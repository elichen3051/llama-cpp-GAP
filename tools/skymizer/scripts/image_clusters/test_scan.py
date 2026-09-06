import io
import unittest

from analysis import (
    connected_groups,
    fingerprint,
    group_stats,
    grouped,
    normalize_question,
    scenario,
    stable_hash,
)
from PIL import Image


def png(image, **kwargs):
    b = io.BytesIO()
    image.save(b, format="PNG", **kwargs)
    return b.getvalue()


def row(uid, images, question):
    return {
        "uid": uid,
        "images": [{"exact_sha256": i} for i in images],
        "question_normalized": question,
        "ordered_image_list_sha256": stable_hash(images),
        "unordered_image_multiset_sha256": stable_hash(sorted(images)),
    }


class Tests(unittest.TestCase):
    def test_encoding_independent(self):
        im = Image.new("RGB", (11, 9), "red")
        a = fingerprint(png(im, compress_level=0))
        b = fingerprint(png(im, compress_level=9))
        self.assertEqual(a["exact_sha256"], b["exact_sha256"])
        self.assertNotEqual(a["encoded_sha256"], b["encoded_sha256"])

    def test_shape_and_rgb(self):
        a = fingerprint(png(Image.new("RGB", (2, 6), "red")))
        b = fingerprint(png(Image.new("RGB", (3, 4), "red")))
        self.assertEqual(a["rgb_pixel_bytes_only_sha1"], b["rgb_pixel_bytes_only_sha1"])
        self.assertNotEqual(a["exact_sha256"], b["exact_sha256"])
        self.assertEqual(
            fingerprint(png(Image.new("L", (2, 2), 0)))["exact_sha256"],
            fingerprint(png(Image.new("RGB", (2, 2), 0)))["exact_sha256"],
        )

    def test_phash_collision_not_identity(self):
        a = fingerprint(png(Image.new("RGB", (10, 10), "red")))
        b = fingerprint(png(Image.new("RGB", (10, 10), "blue")))
        self.assertEqual(a["phash"], b["phash"])
        self.assertNotEqual(a["exact_sha256"], b["exact_sha256"])

    def test_partial_overlap_and_order(self):
        rs = [
            row("1", ["A", "B"], "Q1"),
            row("2", ["A", "C"], "Q2"),
            row("3", ["B", "A"], "Q1"),
        ]
        self.assertEqual(
            group_stats(grouped(rs, "ordered_image_list"))["duplicate_groups"], 0
        )
        self.assertEqual(
            group_stats(grouped(rs, "unordered_image_multiset"))["duplicate_groups"], 1
        )
        self.assertEqual(
            group_stats(grouped(rs, "exact_image"))[
                "distinct_textual_question_clusters"
            ],
            1,
        )
        self.assertEqual(len(connected_groups(rs)), 1)

    def test_repeated_image_not_duplicate_row(self):
        rs = [row("1", ["A", "A"], "Q1")]
        self.assertEqual(group_stats(grouped(rs, "exact_image"))["duplicate_groups"], 0)

    def test_blank_question_unknown(self):
        rs = [row("1", ["A"], ""), row("2", ["A"], "")]
        s = group_stats(grouped(rs, "exact_image"))
        self.assertEqual(s["distinct_textual_question_clusters"], 0)
        self.assertEqual(s["duplicate_groups_with_unknown_question_identity"], 1)

    def test_transitive_components(self):
        rs = [
            row("1", ["A"], "1"),
            row("2", ["A", "B"], "2"),
            row("3", ["B"], "3"),
            row("4", ["C"], "4"),
        ]
        self.assertEqual(sorted(map(len, connected_groups(rs).values())), [1, 3])

    def test_unequal_size_design_effect(self):
        s = scenario([1, 3], 1)
        self.assertEqual(s["design_effect"], 2.5)
        self.assertEqual(s["hypothetical_n_eff"], 1.6)
        self.assertEqual(scenario([1, 3], 0)["hypothetical_n_eff"], 4)

    def test_normalization(self):
        self.assertEqual(normalize_question(" a\n b "), "a b")


class ScannerIntegrationTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, corrupt=False):
        import hashlib
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        red = png(Image.new("RGB", (12, 10), "red"))
        blue = png(Image.new("RGB", (12, 10), "blue"))

        def prepared(number, images, question):
            return {
                "source": "demo",
                "item_id": f"item-{number}",
                "origin_id": number,
                "question": question,
                "images": [{"bytes": b, "path": None} for b in images],
                "num_images": len(images),
                "category": "demo",
                "ref_answer": "answer",
            }

        a = prepared(1, [red], "first question")
        b = prepared(2, [blue], "second question")
        c = prepared(3, [red, red], "third question")
        entries = []
        for name, part, rs in [
            ("demo-subsample-100", 0, [a, b]),
            ("demo-subsample-500", 0, [a, b]),
            ("demo-subsample-500", 1, [c]),
        ]:
            path = self.root / f"{name}-{part}.parquet"
            pq.write_table(pa.Table.from_pylist(rs), path)
            entries.append(
                {
                    "config": name,
                    "split": "train",
                    "remote_path": f"{name}/train-{part:05d}.parquet",
                    "local_cache_path": str(path),
                    "rows": len(rs),
                    "size": path.stat().st_size,
                    "sha256": "0" * 64
                    if corrupt
                    else hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "repo": "test/prepared",
                    "revision": "a" * 40,
                    "selected_files": entries,
                }
            )
        )
        return path

    def run_scan(self, manifest, output):
        import subprocess
        import sys
        from pathlib import Path

        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("scan.py")),
                "--input-manifest",
                str(manifest),
                "--out",
                str(output),
                "--workers",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_multishard_and_expected_prefix(self):
        import csv
        import json

        out = self.root / "out"
        result = self.run_scan(self.manifest(), out)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        status = json.loads((out / "status.json").read_text())
        self.assertEqual(status["status"], "complete")
        rows = [json.loads(s) for s in (out / "rows.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({r["uid"] for r in rows}), 5)
        self.assertEqual(
            [r["row_index"] for r in rows if r["config"].endswith("-500")], [0, 1, 2]
        )
        summary = json.loads((out / "summary.json").read_text())["configs"]
        s = summary["demo-subsample-500"]
        self.assertEqual(s["exact_image"]["duplicate_groups"], 1)
        self.assertEqual(s["exact_image"]["rows_in_duplicate_groups"], 2)
        self.assertEqual(s["within_row_repeated_image_rows"], 1)
        self.assertEqual(
            summary["demo-subsample-100"]["exact_image"]["duplicate_groups"], 0
        )
        self.assertEqual(
            summary["demo-subsample-100"]["phash_nonexact_candidates"]["groups"], 1
        )
        pair = json.loads((out / "overlap_100_500.json").read_text())[0]
        self.assertTrue(pair["ordered_prefix_equal"])
        self.assertEqual(pair["tail_rows_sharing_any_image_with_prefix"], 1)
        with (out / "row_cluster_mapping.csv").open() as f:
            mapping = list(csv.DictReader(f))
        self.assertEqual(len(mapping), 5)
        self.assertEqual(len({r["uid"] for r in mapping}), 5)
        self.assertTrue((out / "scripts" / "analysis.py").is_file())

    def test_integrity_failure_is_not_complete(self):
        import json

        out = self.root / "out"
        result = self.run_scan(self.manifest(corrupt=True), out)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            json.loads((out / "status.json").read_text())["status"], "failed"
        )
        self.assertIn("SHA256 mismatch", (out / "error.log").read_text())
        self.assertFalse((out / "REPORT.md").exists())

    def test_existing_output_is_preserved(self):
        out = self.root / "out"
        out.mkdir()
        (out / "sentinel").write_text("keep")
        result = self.run_scan(self.manifest(), out)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((out / "sentinel").read_text(), "keep")
        self.assertFalse((out / "status.json").exists())

    def test_interrupt_records_state(self):
        import contextlib
        import io
        import json
        from unittest.mock import patch

        import scan

        out = self.root / "interrupted"
        manifest = self.manifest()
        with (
            patch("analysis.run", side_effect=KeyboardInterrupt),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = scan.main(["--input-manifest", str(manifest), "--out", str(out)])
        self.assertEqual(status, 130)
        self.assertEqual(
            json.loads((out / "status.json").read_text())["status"], "interrupted"
        )
        self.assertFalse((out / "REPORT.md").exists())

    def test_alpha_difference_is_preserved(self):
        a = fingerprint(png(Image.new("RGBA", (3, 3), (255, 0, 0, 0))))
        b = fingerprint(png(Image.new("RGBA", (3, 3), (255, 0, 0, 255))))
        self.assertEqual(a["exact_sha256"], b["exact_sha256"])
        self.assertNotEqual(a["rgba_sha256"], b["rgba_sha256"])

    def test_offline_manifest_requires_pin_size_and_digest(self):
        import json

        for defect in (
            "unpinned",
            "missing_size",
            "missing_digest",
            "invalid_digest",
            "bool_size",
        ):
            with self.subTest(defect=defect):
                path = self.manifest()
                metadata = json.loads(path.read_text())
                entry = metadata["selected_files"][0]
                if defect == "unpinned":
                    metadata["revision"] = "main"
                elif defect == "missing_size":
                    entry.pop("size")
                elif defect == "missing_digest":
                    entry.pop("sha256")
                elif defect == "invalid_digest":
                    entry["sha256"] = "not-a-sha256"
                else:
                    entry["size"] = True
                path.write_text(json.dumps(metadata))
                out = self.root / defect
                result = self.run_scan(path, out)
                self.assertNotEqual(result.returncode, 0, defect)
                self.assertEqual(
                    json.loads((out / "status.json").read_text())["status"], "failed"
                )

    def test_nullable_categories_keep_rows_and_missing_bucket(self):
        import csv
        import hashlib
        import json
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq

        manifest = self.manifest()
        metadata = json.loads(manifest.read_text())
        for entry in metadata["selected_files"]:
            path = Path(entry["local_cache_path"])
            rows = pq.read_table(path).to_pylist()
            rows[0]["category"] = None
            pq.write_table(pa.Table.from_pylist(rows), path)
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            entry["size"] = path.stat().st_size
        manifest.write_text(json.dumps(metadata))
        out = self.root / "nullable"
        result = self.run_scan(manifest, out)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with (out / "category_concentration.csv").open() as stream:
            categories = list(csv.DictReader(stream))
        subset = [r for r in categories if r["config"].endswith("-100")]
        self.assertEqual(
            {r["category"]: int(r["rows"]) for r in subset}, {"(missing)": 1, "demo": 1}
        )
        raw = [
            json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()
        ]
        self.assertIsNone(raw[0]["category"])


if __name__ == "__main__":
    unittest.main()
