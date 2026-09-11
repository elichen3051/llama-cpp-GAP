import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from replace_subsamples import (
    STANDARD,
    VISION,
    checked_plan,
    prepare,
    publish,
    validate_pair,
    verify_publication,
)
from replacement import ImageIndex, exact_row_digest, repair_units, row_digest
from scan import file_sha256, save_json


def item(key, images, category="a", seed=1234):
    return {
        "source": "demo",
        "item_id": str(key),
        "origin_id": int(key),
        "sample_seed": seed,
        "question": "question " + str(key),
        "images": [{"bytes": b, "path": None} for b in images],
        "num_images": len(images),
        "ref_answer": "a",
        "category": category,
    }


def signature(value):
    return {"exact_sha256": str(value), "phash": format(value, "016x")}


class FakeFingerprints:
    def unit(self, unit):
        return {
            view: [
                signature(int.from_bytes(im["bytes"], "big") * 65537)
                for im in row["images"]
            ]
            for view, row in unit.items()
        }


def describe(unit):
    return {
        "category": next(iter(unit.values()))["category"],
        "task": "",
        "dataset_name": "",
    }


class SelectionTests(unittest.TestCase):
    def test_view_namespace_and_phash_boundary(self):
        index = ImageIndex(4)
        index.add({"std": [signature(0)]}, 0)
        self.assertEqual(index.conflict({"std": [signature(15)]})["distance"], 4)
        self.assertIsNone(index.conflict({"std": [signature(31)]}))
        self.assertIsNone(index.conflict({"vision": [signature(0)]}))
        self.assertEqual(index.conflict({"std": [signature(0)]})["kind"], "exact_rgb")

    def test_reserve_survivors_and_partial_image_overlap(self):
        def unit(key, images, category="a"):
            return {"v": item(key, images, category)}

        original = [unit(0, [b"A", b"B"]), unit(1, [b"B", b"D"]), unit(2, [b"C"])]
        pool = original + [unit(3, [b"C"]), unit(4, [b"E", b"E"], "b")]
        result, changes, diagnostics = repair_units(
            original, pool, "demo", 1234, 0, FakeFingerprints(), describe
        )
        self.assertEqual([r["v"]["item_id"] for r in result], ["0", "4", "2"])
        self.assertEqual([c["slot"] for c in changes], [1])
        self.assertIn("3", diagnostics["rejected_candidates"])
        self.assertEqual(changes[0]["stratum_tier"], 3)
        self.assertEqual(
            exact_row_digest(result[2]["v"]), exact_row_digest(original[2]["v"])
        )
        repeated = repair_units(
            original, pool, "demo", 1234, 0, FakeFingerprints(), describe
        )
        self.assertEqual(result, repeated[0])

    def test_pool_exhaustion_does_not_shrink(self):
        original = [{"v": item(i, [b"A"])} for i in range(2)]
        with self.assertRaisesRegex(ValueError, "cannot fill"):
            repair_units(
                original, original, "demo", 1234, 0, FakeFingerprints(), describe
            )

    def test_source_digest_ignores_only_seed_and_paths(self):
        a = item(0, [b"A"])
        b = {**a, "sample_seed": None, "images": [{"bytes": b"A", "path": "old.png"}]}
        self.assertEqual(row_digest(a), row_digest(b))
        self.assertNotEqual(exact_row_digest(a), exact_row_digest(b))
        self.assertNotEqual(row_digest(a), row_digest({**b, "ref_answer": "wrong"}))

    def test_pairing_rejects_reordered_ids(self):
        a, b = item(0, [b"A"]), item(1, [b"B"])
        with self.assertRaisesRegex(ValueError, "MMMU view mismatch"):
            validate_pair([a, b], [b, a])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "prepared"

    def inputs(self):
        rng = np.random.default_rng(13)
        images = []
        for _ in range(1006):
            stream = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                stream, format="PNG"
            )
            images.append(stream.getvalue())
        entries, card_info = [], []
        for v, view in enumerate((STANDARD, VISION)):
            pool = [item(i, [images[i + 503 * v]], seed=None) for i in range(503)]
            pool[2 if v == 0 else 6]["images"] = pool[0 if v == 0 else 5]["images"]
            sampled = [{**row, "sample_seed": 1234} for row in pool[:500]]
            schema = pa.Table.from_pylist(sampled).schema.with_metadata(
                {b"test": b"keep"}
            )
            for config, rows in (
                (view, pool),
                (view + "-subsample-500", sampled),
                (view + "-subsample-100", sampled[:100]),
            ):
                path = self.root / (config + ".parquet")
                table = pa.Table.from_pylist(rows, schema=schema)
                pq.write_table(table, path)
                entries.append(
                    {
                        "config": config,
                        "split": "train",
                        "remote_path": config + "/train-00000-of-00001.parquet",
                        "local_cache_path": str(path),
                        "rows": len(rows),
                        "size": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
                card_info.append(
                    {
                        "config_name": config,
                        "download_size": path.stat().st_size,
                        "dataset_size": table.nbytes,
                        "splits": [
                            {
                                "name": "train",
                                "num_examples": len(rows),
                                "num_bytes": table.nbytes,
                            }
                        ],
                    }
                )
        path = self.root / "inputs.json"
        save_json(
            path,
            {
                "repo": "test/prepared",
                "revision": "a" * 40,
                "source_pools": {STANDARD: STANDARD, VISION: VISION},
                "selected_files": entries,
            },
        )
        import yaml

        self.card = self.root / "README.md"
        self.card.write_text(
            "---\n"
            + yaml.safe_dump({"dataset_info": card_info})
            + "---\nTest dataset\n"
        )
        return SimpleNamespace(
            inputs=path,
            strata=None,
            out=self.out,
            seed=1234,
            phash_distance=4,
            workers=1,
        )

    def test_paired_prepare_preserves_slots_prefixes_schema_and_sources(self):
        args = self.inputs()
        with patch("huggingface_hub.hf_hub_download", return_value=str(self.card)):
            self.assertEqual(prepare(args), 0)
        manifest = json.loads((self.out / "draft-manifest.json").read_text())
        final = {
            e["config"]: pq.read_table(e["local_cache_path"]).to_pylist()
            for e in manifest["selected_files"]
        }
        for view in (STANDARD, VISION):
            original = pq.read_table(
                self.root / (view + "-subsample-500.parquet")
            ).to_pylist()
            large, small = (
                final[view + "-subsample-500"],
                final[view + "-subsample-100"],
            )
            self.assertEqual(len(large), 500)
            self.assertEqual(small, large[:100])
            changed = [i for i, (a, b) in enumerate(zip(original, large)) if a != b]
            self.assertEqual(changed, [2, 6])
            self.assertEqual(large[0], original[0])
            table = pq.read_table(
                self.out
                / "upload"
                / (view + "-subsample-500")
                / "train-00000-of-00001.parquet"
            )
            self.assertEqual(table.schema.metadata, {b"test": b"keep"})
        validate_pair(
            final[STANDARD + "-subsample-500"], final[VISION + "-subsample-500"]
        )
        original_manifest = json.loads(args.inputs.read_text())
        for entry in original_manifest["selected_files"]:
            self.assertEqual(file_sha256(entry["local_cache_path"]), entry["sha256"])
        with self.assertRaisesRegex(ValueError, "new or empty"):
            prepare(args)

    def test_invalid_source_row_fails_before_publication(self):
        args = self.inputs()
        manifest = json.loads(args.inputs.read_text())
        entry = manifest["selected_files"][0]
        path = Path(entry["local_cache_path"])
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["question"] = "wrong source question"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        entry.update(size=path.stat().st_size, sha256=file_sha256(path))
        save_json(args.inputs, manifest)
        with self.assertRaisesRegex(ValueError, "does not match source"):
            prepare(args)
        self.assertEqual(
            json.loads((self.out / "status.json").read_text())["status"], "failed"
        )
        self.assertFalse((self.out / "publication-plan.json").exists())

    def plan(self):
        upload = self.out / "upload" / "demo-subsample-500"
        upload.mkdir(parents=True)
        file = upload / "train-00000-of-00001.parquet"
        file.write_bytes(b"reviewed bytes")
        plan = {
            "dataset": "test/prepared",
            "parent_revision": "a" * 40,
            "files": [
                {
                    "path": str(file.relative_to(self.out / "upload")),
                    "size": file.stat().st_size,
                    "sha256": file_sha256(file),
                }
            ],
            "deletions": [],
        }
        save_json(self.out / "publication-plan.json", plan)
        save_json(
            self.out / "draft-manifest.json",
            {**plan, "selected_files": [{"config": "demo-subsample-500"}]},
        )
        review = self.root / "review.json"
        save_json(
            review,
            {
                "status": "approved",
                "publication_plan_sha256": file_sha256(
                    self.out / "publication-plan.json"
                ),
            },
        )
        return SimpleNamespace(out=self.out, review=review), plan, file

    def test_publish_rejects_changed_content_review_and_head(self):
        args, plan, file = self.plan()
        checked_plan(self.out, args.review)
        with (
            patch("huggingface_hub.HfApi") as api,
            patch("subprocess.check_output", return_value="codecommit"),
        ):
            api.return_value.dataset_info.return_value.sha = "b" * 40
            with self.assertRaisesRegex(ValueError, "HEAD changed"):
                publish(args)
            api.return_value.create_commit.assert_not_called()
        file.write_bytes(b"tampered bytes")
        with self.assertRaisesRegex(ValueError, "content changed"):
            checked_plan(self.out, args.review)
        save_json(
            args.review, {"status": "approved", "publication_plan_sha256": "0" * 64}
        )
        with self.assertRaisesRegex(ValueError, "independent approval"):
            checked_plan(self.out, args.review)

    def test_publish_binds_parent_verifies_scope_and_does_not_repeat(self):
        args, plan, _ = self.plan()
        changed = plan["files"][0]
        old_source = SimpleNamespace(
            rfilename="source/train.parquet",
            size=8,
            blob_id="unchanged",
            lfs=SimpleNamespace(sha256="d" * 64),
        )
        old_target = SimpleNamespace(
            rfilename=changed["path"],
            size=3,
            blob_id="old",
            lfs=SimpleNamespace(sha256="c" * 64),
        )
        new_target = SimpleNamespace(
            rfilename=changed["path"],
            size=changed["size"],
            blob_id="new",
            lfs=SimpleNamespace(sha256=changed["sha256"]),
        )
        before = SimpleNamespace(sha="a" * 40, siblings=[old_source, old_target])
        after = SimpleNamespace(sha="b" * 40, siblings=[old_source, new_target])
        with (
            patch("huggingface_hub.HfApi") as api,
            patch("subprocess.check_output", return_value="codecommit"),
        ):
            api.return_value.dataset_info.side_effect = (
                lambda repo, revision=None, **kw: (
                    after if revision == "b" * 40 else before
                )
            )
            api.return_value.create_commit.return_value = SimpleNamespace(
                oid="b" * 40, commit_url="url"
            )
            self.assertEqual(publish(args), 0)
            self.assertEqual(
                api.return_value.create_commit.call_args.kwargs["parent_commit"],
                "a" * 40,
            )
            self.assertEqual(publish(args), 0)
            self.assertEqual(api.return_value.create_commit.call_count, 1)
            receipt = json.loads((self.out / "publication-receipt.json").read_text())
            self.assertEqual(receipt["status"], "published_verified")
            self.assertEqual(receipt["verified_unchanged_files"], 1)
            after.siblings.append(SimpleNamespace(rfilename="unexpected"))
            with self.assertRaisesRegex(ValueError, "paths differ"):
                verify_publication(self.out, plan, "b" * 40)

    def test_ambiguous_publication_stops_retry(self):
        args, plan, _ = self.plan()
        with (
            patch("huggingface_hub.HfApi") as api,
            patch("subprocess.check_output", return_value="codecommit"),
        ):
            api.return_value.dataset_info.return_value.sha = "a" * 40
            api.return_value.create_commit.side_effect = ConnectionError(
                "lost response"
            )
            with self.assertRaises(ConnectionError):
                publish(args)
            with self.assertRaisesRegex(ValueError, "may have reached"):
                publish(args)
            self.assertEqual(api.return_value.create_commit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
