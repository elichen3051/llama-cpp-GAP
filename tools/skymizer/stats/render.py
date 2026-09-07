# stats/render.py -- Markdown rendering of a result dict (pure formatting), plus the inputs /
# execution metadata builders that feed it.
import argparse
import json
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from stats.contracts import SCHEMA_VERSION
from stats.inference import _format_confidence_level
from stats.tokens import _CRITICAL_SAMPLE_TABLE_ROWS

# Rendering
# --------------------------------------------------------------------------- #
def _fmt_signed(v):  return "—" if v is None else f"{float(v):+.6f}"
def _fmt_plain(v):   return "—" if v is None else f"{float(v):.6f}"
def _fmt_signed_ci(ci): return f"[{_fmt_signed(ci.get('lower'))}, {_fmt_signed(ci.get('upper'))}]"
def _fmt_ratio_value(v): return f"× {float(v):.6f}"
def _fmt_ratio_ci(ci):   return f"[× {float(ci['lower']):.6f}, {float(ci['upper']):.6f}]"

_WEIGHTING_KEYS = (("item", "item_weighted"), ("token", "token_weighted"))
_DERIVED_METRIC_LABELS = {
    "ppl": "ppl (a, b = exp(nll))",
    "ppl_ratio": "ppl_ratio (≡ exp(nll Δ))",
    "rms_dp": "rms_dp (a, b = √mean(mse_dp))",
}
_DEFAULT_HIDDEN_TABLE_METRICS = {"rms_dp"}


def _shared_meta_field(metas: Sequence[Mapping | None], field: str):
    """Return the first non-None value of `field` across metas.

    check_meta_alignment guarantees the three dirs agree on every
    _META_MUST_MATCH field when 2+ metas are present, so picking the first
    non-None value is safe -- mismatches fail earlier with AlignmentError."""
    for m in metas:
        if m and m.get(field) is not None:
            return m[field]
    return None


def build_inputs_metadata(
    ref_meta: Mapping | None,
    a_meta: Mapping | None,
    b_meta: Mapping | None,
    ref_label: str,
    a_label: str,
    b_label: str,
) -> dict[str, Any]:
    """Pack the per-side labels/paths + shared dataset fields into
    a single dict for both JSON output and the ## Inputs markdown block."""
    def model_path(m):
        return m["model"] if (m and m.get("model")) else None
    metas = (ref_meta, a_meta, b_meta)
    return {
        "reference":   {"label": ref_label, "model_path": model_path(ref_meta)},
        "candidate_a": {"label": a_label,   "model_path": model_path(a_meta)},
        "candidate_b": {"label": b_label,   "model_path": model_path(b_meta)},
        "kind":    _shared_meta_field(metas, "kind") or (
            "vlm_logits" if any(m is not None for m in metas) else None),
        "dataset": _shared_meta_field(metas, "dataset"),
        "subset":  _shared_meta_field(metas, "subset"),
        "split":   _shared_meta_field(metas, "split"),
        "sort_by": _shared_meta_field(metas, "sort_by"),
        "sort_desc": bool(_shared_meta_field(metas, "sort_desc")),
        # storage dtype of the logits dumps (None for VLMK metric dirs, which
        # store no logits): drives the report's "Logits format" line
        "logits_dtype": _shared_meta_field(metas, "logits_dtype"),
    }


def _format_dataset_line(inputs: Mapping[str, Any]) -> str | None:
    """`<dataset> / <subset> / <split> (sort_by=<sort_by>)`; skip empty parts."""
    ds = inputs.get("dataset")
    if not ds:
        return None
    parts = [ds]
    for f in ("subset", "split"):
        v = inputs.get(f)
        if v:
            parts.append(v)
    line = " / ".join(parts)
    sb = inputs.get("sort_by")
    if sb:
        line += f" (sort_by={sb}{', desc' if inputs.get('sort_desc') else ''})"
    return line


def _sorted_prefix_warning(inputs: Mapping[str, Any], n_items,
                           start: int | None, end: int | None) -> str | None:
    """The item set is a PREFIX of a sorted order, not a random sample of the
    dataset -- say so where the dataset is named.

    The collectors sort by num_images by default, so "the first N rows" is the
    fewest-image tail. A report that names the dataset with no qualifier
    invites the reader to generalize from a systematically atypical slice, and
    the effect being measured is a VISION one."""
    sb = inputs.get("sort_by")
    if not sb:
        return None
    windowed = bool(start) or (end is not None and end != -1)
    which = "largest" if inputs.get("sort_desc") else "smallest"
    detail = (f"the items are a WINDOW of that order (--start {start or 0}, "
              f"--end {end})" if windowed else
              f"the {n_items} item(s) are a PREFIX of that order")
    return (f"- ⚠ Items were collected sorted by `{sb}`, and {detail} — the "
            f"{which}-`{sb}` slice of the dataset, not a random sample of it. "
            "Generalize to the dataset only with that in mind.")


def _jsonable_arg_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable_arg_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_arg_value(v) for v in value]
    return value


def _meminfo_bytes() -> dict[str, int | None]:
    out = {"ram_total_bytes": None, "ram_available_bytes": None}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, rest = line.split(":", 1)
                if key == "MemTotal":
                    out["ram_total_bytes"] = int(rest.split()[0]) * 1024
                elif key == "MemAvailable":
                    out["ram_available_bytes"] = int(rest.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def _cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") or line.startswith("Hardware"):
                    return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def _system_metadata() -> dict[str, Any]:
    mem = _meminfo_bytes()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 1,
        "cpu_model": _cpu_model_name(),
        **mem,
    }


def _reproducible_report_env() -> bool:
    """True when SKYMIZER_REPRODUCIBLE_REPORT is set to a truthy value; makes
    --omit-host-metadata the default so wrapper scripts can pin byte-stable
    reports without threading a flag through every call site."""
    v = os.environ.get("SKYMIZER_REPRODUCIBLE_REPORT", "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def build_execution_metadata(
    args: argparse.Namespace,
    argv: Sequence[str],
    *,
    device: str,
    jobs: int,
    auto_jobs_estimate: Mapping[str, Any] | None = None,
    omit_host_metadata: bool = False,
) -> dict[str, Any]:
    """With omit_host_metadata the host-VOLATILE fields are dropped -- the
    system block (MemAvailable changes between two back-to-back runs and is
    rendered into the markdown) and the auto-jobs estimate derived from it --
    so two runs of the same command produce byte-identical report + JSON.
    That byte-diff is the refactor plan's primary oracle; everything kept
    here (command, argv, args, resolved device/jobs) is a pure function of
    the invocation."""
    argv = [str(x) for x in argv]
    resolved = {"device": device, "jobs": jobs}
    if auto_jobs_estimate is not None and not omit_host_metadata:
        resolved["auto_jobs_estimate"] = _jsonable_arg_value(auto_jobs_estimate)
    out = {
        "command": shlex.join(argv),
        "argv": argv,
        "args": _jsonable_arg_value(vars(args)),
        "resolved": resolved,
    }
    if not omit_host_metadata:
        out["system"] = _system_metadata()
    return out


def _format_bytes_or_unknown(n) -> str:
    if isinstance(n, (int, float)) and n > 0:
        return f"{float(n) / (1024 ** 3):.2f} GiB"
    return "unknown"


def _format_execution_section(execution: Mapping[str, Any]) -> list[str]:
    lines = ["## Execution", ""]
    command = execution.get("command")
    if command:
        lines.extend(["- Command:", "", "```sh", str(command), "```", ""])
    resolved = execution.get("resolved") or {}
    if resolved:
        device = resolved.get("device", "unknown")
        jobs = resolved.get("jobs", "unknown")
        collection = execution.get("metrics_collection") or {}
        if collection:
            gpu_by_candidate = collection.get("gpu_by_candidate") or {}
            gpu_values = set(gpu_by_candidate.values())
            if len(gpu_values) == 1:
                gpu_description = next(iter(gpu_values))
            else:
                gpu_description = "; ".join(
                    f"{role}={gpu_by_candidate.get(role, 'unknown')}"
                    for role in ("candidate-a", "candidate-b")
                )
            lines.append(
                f"- GPU recorded on metrics-collection host: {gpu_description or 'unknown'}")
            lines.append(f"- Paired statistics backend: {device}; jobs={jobs}")
        else:
            lines.append(f"- Resolved backend: {device}; jobs={jobs}")
    system = execution.get("system") or {}
    if system:
        cpu_count = system.get("cpu_count", "unknown")
        cpu_model = system.get("cpu_model") or "unknown"
        lines.append(f"- CPU: {cpu_model} ({cpu_count} logical)")
        lines.append(
            "- RAM: "
            f"{_format_bytes_or_unknown(system.get('ram_total_bytes'))} total, "
            f"{_format_bytes_or_unknown(system.get('ram_available_bytes'))} available"
        )
        if system.get("platform") or system.get("python"):
            lines.append(
                f"- Platform: {system.get('platform', 'unknown')}; "
                f"Python {system.get('python', 'unknown')}"
            )
    args = execution.get("args") or {}
    if args:
        lines.extend(["", "Arguments:", ""])
        rows = [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v]
                for k, v in args.items()]
        lines.append(_render_table(["argument", "value"], rows))
    lines.append("")
    return lines


def _render_table(headers, rows, *, aligns=None):
    """Render a GitHub-Flavored Markdown table.

    `aligns` is a per-column list of 'left' / 'right' / 'center' (default all
    left). Numeric columns look much better right-aligned in the rendered
    output. Cell text is padded to the column width in the raw source so the
    .md is also scannable as plain text -- the GFM renderer ignores the
    padding."""
    n = len(headers)
    aligns = list(aligns) if aligns is not None else ["left"] * n
    if len(aligns) != n:
        raise ValueError(f"aligns must have one entry per column ({n}); got {len(aligns)}")
    widths = [max(3, max(len(str(r[i])) for r in [headers, *rows])) for i in range(n)]

    def _cell(value, width, align):
        s = str(value)
        return s.rjust(width) if align == "right" else s.ljust(width)

    def _row(cells):
        return "| " + " | ".join(_cell(c, widths[i], aligns[i]) for i, c in enumerate(cells)) + " |"

    sep_cells = []
    for i, a in enumerate(aligns):
        w = widths[i]
        if a == "right":
            sep_cells.append("-" * (w - 1) + ":")
        elif a == "center":
            sep_cells.append(":" + "-" * (w - 2) + ":")
        else:
            sep_cells.append("-" * w)
    sep = "| " + " | ".join(sep_cells) + " |"

    return "\n".join([_row(headers), sep, *(_row(r) for r in rows)])


def _resolve_display_weightings(display_weighting: str):
    if display_weighting == "both":
        return _WEIGHTING_KEYS
    if display_weighting == "item":
        return (("item", "item_weighted"),)
    if display_weighting == "token":
        return (("token", "token_weighted"),)
    raise ValueError(f"display_weighting must be item|token|both; got {display_weighting!r}")


def _section_title(a_label, b_label, n_items, iters, conf_pct, num_eval_tokens,
                   ci_method=None, sampling=None) -> list[str]:
    if ci_method == "t":
        how = f"{conf_pct}CI via paired Student-t (df = n - 1, no bootstrap)"
    else:
        how = f"{conf_pct}CI via {iters} paired-bootstrap iters"
    lines = [
        f"# Paired comparison: {a_label} (a) vs {b_label} (b)",
        "",
        f"- Items compared: {n_items}   |   {how}",
        "- Protocol: teacher-forced on the dataset's stored model-generated "
        "answer trajectory; KLD/JSD/EAR over full vocab; answer positions only.",
    ]
    lines.append("- Paired inference is item-weighted only; token-weighted rows are descriptive.")
    if sampling:
        lines[2] = f"- Paired {sampling['unit']} units: {n_items} from {sampling['n_windows']} corpus windows   |   {how}"
        protocol = sampling["protocol"]
        lines[3] = f"- Protocol: fixed text corpus, {protocol['window_size']}-token windows, {protocol['targets_per_window']} teacher-forced targets per window; KLD/JSD/EAR over full vocabulary."
        lines.append("- " + sampling["scope"])
        lines.append("- Item weighting gives each selected group equal weight; token weighting pools its scored targets.")
        if sampling.get("block_windows"):
            lines.append(f"- Block size: {sampling['block_windows']} consecutive original windows; the final partial block is retained.")
    if num_eval_tokens != -1:
        lines.append(f"- Compare-time num_eval_tokens: {num_eval_tokens}")
    lines.append("")
    return lines


def _section_inputs(result, reference_label, n_items) -> list[str]:
    """## Inputs -- model paths, logits-file format, dataset identity.
    Falls back to the legacy "Reference (FP teacher): X" bullet only when the
    caller hasn't populated result["inputs"] (older JSON or direct API users)."""
    lines: list[str] = []
    inputs = result.get("inputs")
    if inputs:
        lines.append("## Inputs")
        lines.append("")
        table_rows = []
        for role_key, role_label in (
            ("reference", "reference"),
            ("candidate_a", "candidate-a"),
            ("candidate_b", "candidate-b"),
        ):
            side = inputs.get(role_key, {})
            label = side.get("label") or "—"
            path = side.get("model_path") or "(collect_meta.json missing)"
            table_rows.append([role_label, label, path])
        lines.append(_render_table(["role", "label", "model path"], table_rows))
        lines.append("")
        if inputs.get("kind"):
            lines.append(f"- Artifact kind: {inputs.get('kind')}")
        logits_dtype = inputs.get("logits_dtype")
        if logits_dtype:
            # only logits-dump dirs carry a storage dtype; VLMK metric-dump
            # reports store no logits (their trailing note says so)
            lines.append(f"- Logits format: dense {logits_dtype}, full vocab")
        ds_line = _format_dataset_line(inputs)
        if ds_line:
            lines.append(f"- Dataset: {ds_line}")
        exec_args = (result.get("execution") or {}).get("args") or {}
        prefix_note = _sorted_prefix_warning(
            inputs, n_items, exec_args.get("start"), exec_args.get("end"))
        if prefix_note:
            lines.append(prefix_note)
        lines.append("")
    else:
        lines.append(f"- Reference (FP teacher): {reference_label}")
        lines.append("")
    return lines


def _section_alignment(
    drops,
    n_matched,
    n_items,
    *,
    allow_interaction: bool = False,
    common_budget_skips=(),
) -> list[str]:
    """Persist any explicit intersection or common budget exclusion."""
    lines: list[str] = []
    has_keyset_drops = bool(drops and any(drops.get(r) for r in drops))
    common_budget_skips = list(common_budget_skips or ())
    if not has_keyset_drops and not common_budget_skips:
        return lines

    lines.append("## Alignment")
    lines.append("")
    if has_keyset_drops:
        shown = n_matched if n_matched is not None else n_items
        total = sum(len(drops[r]) for r in drops)
        lines.append(f"> ⚠ Compared on the {shown}-item intersection. "
                     f"Dropped {total} item(s) not present in all dirs:")
        if allow_interaction:
            lines.append(
                "> ⚠ `--allow-interaction` explicitly enabled this reduced sample.")
        for role in ("reference", "candidate-a", "candidate-b"):
            miss = drops.get(role) or []
            if miss:
                sample = ", ".join(miss[:5])
                more = "" if len(miss) <= 5 else f", … (+{len(miss) - 5})"
                lines.append(f">   - {role} missing {len(miss)}: {sample}{more}")

    if common_budget_skips:
        sample = ", ".join(str(idx) for idx in common_budget_skips[:10])
        more = ("" if len(common_budget_skips) <= 10 else
                f", … (+{len(common_budget_skips) - 10})")
        lines.append(
            f"> ℹ Common SKIP_OVER_BUDGET: {len(common_budget_skips)} "
            f"row(s): {sample}{more}")
    lines.append("")
    return lines


def _section_reference_likelihood(ref_m, reference_label, weightings) -> list[str]:
    """Full-logits reference likelihood at the same target
    tokens. Its own section, not a row in the paired table -- there is
    nothing paired about it (one shared reference, checked for agreement in
    both columns by construction)."""
    lines: list[str] = []
    if ref_m:
        lines.append(f"## Reference corpus likelihood ({reference_label})")
        lines.append("")
        lines.append(_render_table(
            ["weighting", "mean nll_ref (nats)", "PPL(reference)"],
            [[short, _fmt_plain(ref_m["nll"][f"{short}_weighted_mean"]),
              _fmt_plain(ref_m["ppl"][f"{short}_weighted"])]
             for short, _k in weightings],
            aligns=["left", "right", "right"]))
        lines.append("")
        lines.append(
            "Reference likelihood on the same stored target tokens, using uncompressed VLMK NLL values. "
            "The reference is evaluated in each collection; the JSON's `max_abs_side_difference` "
            f"({ref_m.get('max_abs_side_difference', 0.0):.3g}) records the observed agreement.")
        lines.append("")
    return lines


def _verdict_text(decision):
    verdict = decision["verdict"]
    if decision.get("equivalence_established") and verdict != "EQUIVALENT":
        verdict += f"; equivalent at margin {decision['equivalence_margin']:g}"
    return verdict


def _section_verdict_summary(result, weightings) -> list[str]:
    summary_rows = []
    for name, mres in (result.get("metrics") or {}).items():
        if "source_metric" in mres:
            continue
        # --weighting item|token|both must filter this summary too; hard-coding
        # both rows made `--weighting item` print token rows the results table
        # below then omits. Byte-neutral at the "both" default.
        for wshort, wkey in weightings:
            block = mres.get(wkey)
            if not isinstance(block, dict) or block.get("role") is None:
                continue
            role = block["role"]
            if role == "primary":
                scope = "primary (conditional corpus inference)" if result.get("sampling") else "primary (confirmatory)"
            elif block.get("holm_significant"):
                scope = "exploratory (survives Holm)"
            else:
                continue
            summary_rows.append([f"{name} ({wshort})",
                                 _verdict_text(block["decision"]), scope])
    lines: list[str] = []
    if summary_rows:
        lines.append("## Verdict summary")
        lines.append("")
        lines.append(_render_table(["endpoint", "verdict", "inference scope"],
                                   summary_rows,
                                   aligns=["left", "left", "left"]))
        lines.append("")
        lines.append("Item-weighted endpoints not listed here are either "
                     "inconclusive or did not survive the family-wise "
                     "correction; the full grid is below.")
        lines.append("")
    return lines


def _section_results_intro(primary, multiplicity, conf_pct, conditional=False) -> list[str]:
    lines = ["## Results", ""]
    lines.append("verdict reads the paired CI of (b − a); do NOT compare per-model "
                 "means independently.")
    if primary.get("metric"):
        lines.append("")
        lines.append(
            f"**★ = the one confirmatory endpoint** (`{primary['metric']}`, "
            f"{primary['weighting']}-weighted). Its {conf_pct}CI is the "
            "report's claim and spends the whole alpha. Other item-weighted endpoints are "
            "**exploratory**: its `p (Holm)` is the Holm–Bonferroni-adjusted "
            f"p-value across the other {multiplicity.get('family_size', 0)} "
            "cells (`✓` = survives at α, `·` = does not), and only a `✓` row "
            "supports a claim of its own. Reading the raw per-cell verdicts as "
            f"{multiplicity.get('family_size', 0) + 1} independent tests is "
            "what produced a measured family-wise false-positive rate of 0.35 "
            "on exchangeable A/B data.")
    lines.append("")
    lines.append("Token-weighted rows show descriptive means and differences only, without a CI, p-value or paired-test verdict.")
    lines.append("")
    if conditional:
        lines = [line.replace("the one confirmatory endpoint", "the one primary endpoint, conditional on the corpus assumptions") for line in lines]
    return lines


def _results_rows(result, metrics_to_show, weightings, primary_cell) -> list[list[str]]:
    """The rows of the main results table: one per (metric, weighting), the
    derived rows (ppl / ppl_ratio / rms_dp) rendered from their linked block,
    and the candidate-only rows inserted right after `nll` (or at the end
    when nll is absent) -- exactly once."""
    rows: list[list[str]] = []
    candidate_metric_rows_added = False

    def append_candidate_metric_rows() -> None:
        nonlocal candidate_metric_rows_added
        if candidate_metric_rows_added:
            return
        candidate_metric_rows_added = True

        for cname, cmres in (result.get("candidate_metrics") or {}).items():
            unit = cmres.get("unit")
            label = f"{cname} ({unit})" if unit else cname
            for cshort_name, _cblock_key in weightings:
                mean_key = f"{cshort_name}_weighted_mean"
                a_side = cmres.get("candidate_a", {})
                b_side = cmres.get("candidate_b", {})
                rows.append([
                    label,
                    cshort_name,
                    _fmt_plain(a_side.get(mean_key)),
                    _fmt_plain(b_side.get(mean_key)),
                    "—",
                    "—",
                    "—",
                    "candidate-only; not compared",
                ])

    for name, mres in metrics_to_show:
        for short_name, block_key in weightings:
            block = mres.get(block_key)
            if block is None:
                continue
            descriptive = block.get("role") == "descriptive"
            decision = block.get("decision", {})
            verdict = "descriptive only" if descriptive else _verdict_text(decision)
            if "delta_candidate_minus_baseline" in block:
                ci = block.get("ci_delta", {})
                estimate_str = _fmt_signed(block["delta_candidate_minus_baseline"])
                ci_str = _fmt_signed_ci(ci)
                a_mean, b_mean, label = block.get("baseline_mean"), block.get("candidate_mean"), name
            elif name in _DERIVED_METRIC_LABELS and (descriptive or "linked_to" in decision):
                label = _DERIVED_METRIC_LABELS[name]
                if name == "ppl_ratio":
                    a_mean = b_mean = None
                    ci = block.get("ci", {})
                    estimate_str = _fmt_ratio_value(block["estimate"])
                    ci_str = "-" if descriptive else _fmt_ratio_ci(ci)
                elif name == "ppl":
                    a_mean, b_mean = block.get("baseline_ppl"), block.get("candidate_ppl")
                    estimate_str = _fmt_signed(block.get("delta_ppl_b_minus_a"))
                    ci_str = "(see ppl_ratio)"
                else:  # rms_dp
                    a_mean, b_mean = block.get("baseline_rms"), block.get("candidate_rms")
                    estimate_str = _fmt_signed(block.get("delta_rms_b_minus_a"))
                    ci_str = "(see mse_dp)"
            else:
                label = name
                a_mean = b_mean = None
                ci = block.get("ci", block.get("ci_delta", {}))
                estimate_str = _fmt_signed(block.get("estimate"))
                ci_str = _fmt_signed_ci(ci)
            if (name, block_key) == primary_cell:
                label = f"★ {label}"
                p_str = (f"{block['p_value']:.4g} (primary)"
                         if block.get("p_value") is not None else "—")
            elif block.get("p_value_holm") is not None:
                mark = "✓" if block.get("holm_significant") else "·"
                p_str = f"{block['p_value_holm']:.4g} {mark}"
            else:
                p_str = "—"
            if descriptive:
                ci_str = p_str = "-"
            rows.append([label, short_name, _fmt_plain(a_mean), _fmt_plain(b_mean),
                         estimate_str, ci_str, p_str, verdict])
        if name == "nll":
            append_candidate_metric_rows()

    append_candidate_metric_rows()
    return rows


def _section_notes(metrics_to_show, n_items, result_metrics) -> list[str]:
    note_lines = []
    if n_items is not None and n_items < 30:
        note_lines.append(
            f"note: small sample (n_items = {n_items} < 30). No interval "
            "construction reaches its nominal coverage on right-skewed "
            "per-item deltas at this size (measured ~0.90 for a nominal 0.95 "
            "at n = 25, skew 6, for the t and bootstrap-t intervals alike), "
            "and the misses sit on one side: the interval tends to "
            "understate the damage of rare catastrophic items. The real α "
            "is above the stated one.")
    note_lines.append(
        "note: `inconclusive` means the interval contains 0 — NOT that the two "
        "candidates are equivalent. Each such row's `reason` carries the "
        "tightest margin its own interval does rule out; pass "
        "--equivalence-margin to have the primary metric tested for "
        "equivalence at a margin you choose (interval-inclusion TOST).")
    note_lines.append("note: mse_dp is in pp² (percentage-points squared); rms_dp (pp) "
                      "is shown with --show-diagnostic-metrics.")
    note_lines.append("note: ppl = exp(mean nll) per model (item/token-weighted); only the item-weighted ppl_ratio has a paired-test interval.")
    if "ear" in result_metrics:
        note_lines.append(
            "note: ear = Expected Acceptance Rate (arXiv:2605.02404): per position "
            "Σ_v min(p_ref, p_cand) = 1 − TV distance over the full vocab; "
            "EAR 0.99 ⇒ the two models emit the same token 99% of the time under "
            "optimal coupling. Higher is better.")
    return note_lines


def format_comparison_table(
    result: Mapping[str, Any],
    *,
    reference_label: str,
    display_weighting: str = "both",
    show_diagnostic_metrics: bool = False,
    drops: Mapping[str, Sequence[str]] | None = None,
    n_matched: int | None = None,
    num_eval_tokens: int = -1,
) -> str:
    """Render the markdown report. Pure formatting of an already-computed
    `result` (no arithmetic here beyond number formatting); the sections are
    emitted in this fixed order by the `_section_*` helpers above:
    title -> inputs -> execution -> alignment -> reference likelihood ->
    verdict summary -> results (intro, table, notes) -> position strata ->
    pooled per-token distributions."""
    schema = result.get("schema_version", "")
    if schema and schema != SCHEMA_VERSION:
        raise ValueError(f"schema {schema!r} != {SCHEMA_VERSION!r}; rerun paired_compare")

    weightings = _resolve_display_weightings(display_weighting)

    a_label = result.get("model_a_label", "a")
    b_label = result.get("model_b_label", "b")
    n_items = result.get("n_items")
    iters = result.get("bootstrap_iters")
    conf = result.get("confidence_level")
    conf_pct = "" if conf is None else f"{_format_confidence_level(conf)} "
    multiplicity = result.get("multiplicity") or {}
    primary = multiplicity.get("primary_endpoint") or {}
    primary_cell = (primary.get("metric"),
                    f"{primary.get('weighting')}_weighted")

    lines = _section_title(a_label, b_label, n_items, iters, conf_pct, num_eval_tokens,
                           ci_method=result.get("ci_method"), sampling=result.get("sampling"))
    lines += _section_inputs(result, reference_label, n_items)
    execution = result.get("execution")
    if execution:
        lines.extend(_format_execution_section(execution))
    alignment = result.get("alignment") or {}
    lines += _section_alignment(
        drops, n_matched, n_items,
        allow_interaction=bool(alignment.get("allow_interaction")),
        common_budget_skips=alignment.get("common_skipped_over_budget") or ())
    lines += _section_reference_likelihood(result.get("reference_metrics"),
                                           reference_label, weightings)
    lines += _section_verdict_summary(result, weightings)
    lines += _section_results_intro(primary, multiplicity, conf_pct, conditional=bool(result.get("sampling")))

    ci_header = f"{_format_confidence_level(conf)} CI" if conf is not None else "CI"
    headers = ["metric", "weighting", "a mean", "b mean", "b − a (or b ÷ a)",
               ci_header, "p (Holm)", "verdict"]
    metrics_to_show = [
        (name, mres) for name, mres in result.get("metrics", {}).items()
        if show_diagnostic_metrics or name not in _DEFAULT_HIDDEN_TABLE_METRICS
    ]
    rows = _results_rows(result, metrics_to_show, weightings, primary_cell)
    note_lines = _section_notes(metrics_to_show, n_items, result.get("metrics", {}))

    # metric / weighting are labels; a mean / b mean / delta are numeric;
    # CI / verdict are mostly textual — keep them left-aligned.
    metric_aligns = ["left", "left", "right", "right", "right", "left", "right",
                     "left"]
    body = lines + [_render_table(headers, rows, aligns=metric_aligns), ""] + note_lines
    body += _format_position_strata(result.get("position_strata"),
                                    a_label, b_label, conf)
    body += _format_per_item_tails(result.get("per_item_tails"),
                                   a_label, b_label, conf)
    body += _format_pooled_distribution(result.get("pooled_token_distribution"),
                                        a_label, b_label)
    return "\n".join(body) + "\n"


_PER_ITEM_TAIL_LABELS = {"p99": "p99", "p999": "p99.9", "max": "max"}
_PER_ITEM_TAIL_TABLE_ROWS = 10      # markdown cap for the worst-items table; JSON keeps every item


def _fmt_tail_witness(entry: Mapping[str, Any] | None) -> str:
    """"0.4321 @17 tok 151645" -- a per-item tail value, the answer position it
    occurred at (0-based among scored positions, after any compare-time cap)
    and, when known, the target token id there."""
    if not entry:
        return "—"
    s = f"{float(entry['value']):.4g} @{entry['position']}"
    if "target_token" in entry:
        s += f" tok {entry['target_token']}"
    return s


def _format_per_item_tails(tails, a_label: str, b_label: str,
                           conf: float | None) -> list[str]:
    """Per-item p99 / p99.9 / max of the per-token KLD: one exploratory test
    per level (own Holm family) and the worst items with their witnesses."""
    if not tails:
        return []
    level = (_format_confidence_level(conf) if conf is not None else "")
    out = [
        "",
        "## Per-item KLD tails (exploratory — own Holm family)",
        "",
        "Each item's OWN p99 / p99.9 / maximum of its per-token KLD, with the "
        "answer position (and target token id) it occurred at, so a bad tail "
        "can be traced to the token that produced it. The quantile convention "
        "is the pooled ladder's (linear interpolation between neighbouring "
        "order statistics) applied to one item at a time; `max` is exact.",
        "",
        "Across items each level is a per-item scalar and is tested with the "
        "same paired statistics as the mean metrics (unit: the item), but "
        "these rows are **exploratory**: a per-item maximum is a single-token "
        "statistic, far heavier-tailed than a mean, so its interval reaches "
        "nominal coverage later; `p (Holm)` is adjusted within this family "
        "only and never borrows the confirmatory endpoint's α. `on top-2` "
        "counts the items too short for that level to interpolate anywhere "
        "but between their two largest tokens (p99 needs > 100 positions, "
        "p99.9 > 1000) — for them the level is the max in all but name.",
        "",
    ]
    headers = ["metric", "level", "items", "on top-2", f"a = {a_label}",
               f"b = {b_label}", "b − a", f"{level} CI".strip(),
               "p (Holm)", "verdict"]
    rows = []
    for metric, block in tails.get("metrics", {}).items():
        for c in block.get("cells", []):
            label = _PER_ITEM_TAIL_LABELS.get(c["level"], c["level"])
            top2 = ("—" if "n_items_resting_on_top_two" not in c
                    else str(c["n_items_resting_on_top_two"]))
            if "p_value" not in c:
                rows.append([metric, label, str(c["n_items_used"]), top2,
                             "—", "—", "—", "—", "—",
                             c.get("skipped", "not enough items")])
                continue
            mark = "✓" if c.get("holm_significant") else "·"
            rows.append([
                metric, label, str(c["n_items_used"]), top2,
                _fmt_plain(c.get("baseline_mean")),
                _fmt_plain(c.get("candidate_mean")),
                _fmt_signed(c.get("delta_candidate_minus_baseline")),
                _fmt_signed_ci(c.get("ci_delta", {})),
                f"{c['p_value_holm']:.4g} {mark}"
                if c.get("p_value_holm") is not None else "—",
                c["decision"]["verdict"],
            ])
    out.append(_render_table(
        headers, rows,
        aligns=["left", "left", "right", "right", "right", "right", "right",
                "left", "right", "left"]))
    # Worst items: ranked by the larger of the two candidates' maxima, so an
    # item that is catastrophic for EITHER candidate surfaces.
    for metric, block in tails.get("metrics", {}).items():
        items = [it for it in block.get("items", []) if "candidate_a" in it]
        if not items:
            continue
        items.sort(key=lambda it: -max(it["candidate_a"]["max"]["value"],
                                       it["candidate_b"]["max"]["value"]))
        shown = items[:_PER_ITEM_TAIL_TABLE_ROWS]
        out += ["",
                f"Worst {len(shown)} of {len(items)} items by per-item max "
                f"`{metric}` (either candidate); every item is in the JSON "
                "under `per_item_tails`:", ""]
        h = ["item", "positions", f"a max @pos", f"b max @pos", "b − a (max)",
             "a p99 @pos", "b p99 @pos"]
        r = []
        for it in shown:
            who = it.get("item_key", f"item {it['item_index']}")
            r.append([who, str(it["n_positions"]),
                      _fmt_tail_witness(it["candidate_a"]["max"]),
                      _fmt_tail_witness(it["candidate_b"]["max"]),
                      _fmt_signed(it["delta_b_minus_a"]["max"]),
                      _fmt_tail_witness(it["candidate_a"]["p99"]),
                      _fmt_tail_witness(it["candidate_b"]["p99"])])
        out.append(_render_table(h, r, aligns=["left", "right", "left", "left",
                                               "right", "left", "left"]))
    return out


def _format_position_strata(strata, a_label: str, b_label: str,
                            conf: float | None) -> list[str]:
    """The answer-position breakdown. Deliberately its own section, labelled
    exploratory, with its own Holm family."""
    if not strata:
        return []
    level = (_format_confidence_level(conf) if conf is not None else "")
    out = [
        "",
        "## Answer-position strata (exploratory)",
        "",
        "Per-item means restricted to each answer-position range, tested with "
        "the same paired ITEM bootstrap (an item contributes the mean of its "
        "own positions in the bucket; items that never reach a bucket are "
        "dropped from it).",
        "",
        "This exists because the effect is not spread evenly over the answer: "
        "the mmproj F16-vs-Q8_0 signal measured on this data sits in the "
        "**first ~32 answer tokens** and decays more than 10× after, while "
        "the default eval cap is 1024 — so a single item-mean over all "
        "positions dilutes a vision effect by roughly 30×. Capping the run "
        "shorter throws the rest of the data away; stratifying keeps it.",
        "",
        "The strata were chosen from prior measurements on this data, so every "
        "row here is **exploratory**: `p (Holm)` is adjusted within this "
        "family only, and these rows never carry the confirmatory endpoint's "
        "α.",
        "",
    ]
    headers = ["metric", "positions", "items", f"a = {a_label}",
               f"b = {b_label}", "b − a", f"{level} CI".strip(),
               "p (Holm)", "verdict"]
    rows = []
    for metric, cells in strata.get("metrics", {}).items():
        for c in cells:
            if "p_value" not in c:
                rows.append([metric, c["bucket"], str(c["n_items_used"]),
                             "—", "—", "—", "—", "—",
                             c.get("skipped", "not enough items")])
                continue
            mark = "✓" if c.get("holm_significant") else "·"
            rows.append([
                metric, c["bucket"], str(c["n_items_used"]),
                _fmt_plain(c.get("baseline_mean")),
                _fmt_plain(c.get("candidate_mean")),
                _fmt_signed(c.get("delta_candidate_minus_baseline")),
                _fmt_signed_ci(c.get("ci_delta", {})),
                f"{c['p_value_holm']:.4g} {mark}"
                if c.get("p_value_holm") is not None else "—",
                c["decision"]["verdict"],
            ])
    out.append(_render_table(
        headers, rows,
        aligns=["left", "left", "right", "right", "right", "right", "left",
                "right", "left"]))
    return out


_POOLED_LADDER_LABELS = {
    "max": "Maximum", "p999": "99.9%", "p99": "99.0%", "p95": "95.0%",
    "p90": "90.0%", "median": "Median", "p10": "10.0%", "p05": " 5.0%",
    "p01": " 1.0%", "p001": " 0.1%", "min": "Minimum",
}

_POOLED_METRIC_HEADINGS = {
    "dp": ("Δp at the target token",
           "p_cand(target) − p_ref(target) in percentage points, SIGNED. "
           "The full ladder including the MEDIAN is what shows a systematic "
           "over- or under-confidence; `mse_dp` and `rms_dp` are unsigned by "
           "construction and cannot. This is llama-perplexity's Δp table"),
    "kld": ("KLD",
            "forward KL(p_ref ‖ p_cand) at each scored answer position, nats. "
            "Lower is better, so the degradation tail is the HIGH end: "
            "99.0% / 99.9% / Maximum. At the BOTTOM of the ladder the values "
            "are at the floating-point noise floor — where the two "
            "distributions are all but identical the double-precision sum of "
            "signed per-slot terms can land a few ulp below zero, and the "
            "float32 store keeps it there. A tiny negative minimum is that "
            "floor, not a broken metric; it is left unclamped because "
            "clamping would hide it"),
    "ear": ("EAR",
            "Σ_v min(p_ref, p_cand) = 1 − TV at each scored answer position. "
            "HIGHER is better, so the degradation tail is the LOW end: "
            "1.0% / 0.1% / Minimum — a literal \"maximum EAR\" sits at ~1.0 "
            "for any candidate worth measuring and carries no information"),
}


def _fmt_witness(w: Mapping[str, Any] | None) -> str:
    """"key@pos" (or "item 7@pos") — the pooled token a ladder value came
    from. `position` is the 0-based index among that item's scored ANSWER
    positions, after any compare-time --num-eval-tokens cap."""
    if not w:
        return "—"
    who = w.get("item_key") or f"item {w.get('item_index')}"
    return f"{who}@{w.get('position')}"


def _format_pooled_distribution(pooled, a_label: str, b_label: str) -> list[str]:
    """The descriptive per-token distribution ladders, one table per metric.

    Deliberately OUTSIDE the verdict table and with no CI column: these are
    observed statistics of the pooled token multiset, not paired estimates
    with an inferential guarantee. See _pooled_ladder for why the pooled max
    in particular cannot carry a nonparametric bootstrap interval.
    """
    if not pooled:
        return []
    first = next(iter(pooled.values()))["candidate_a"]
    out = [
        "",
        "## Per-token distributions (pooled tokens — NO bootstrap, NO CI)",
        "",
        f"{first['n_tokens']} token(s) from {first['n_items']} item(s), "
        "flattened across items into ONE multiset — the llama-perplexity "
        "`--kl-divergence` scale and convention (linear interpolation between "
        "neighbouring order statistics).",
        "",
        "These rows are **descriptive statistics with no confidence interval, "
        "no bootstrap and no verdict**. The pooled maximum has no consistent "
        "nonparametric bootstrap — a resample can only contain a subset of "
        "the observed items, so its replicate distribution is degenerate "
        "(measured coverage 0.86–0.89 against a nominal 0.95, and *worsening* "
        "as items are added) — and the far quantiles rest on too few tokens "
        "to support one either. Confirmatory inference lives on the mean "
        "metrics above; the bold rows are the degradation tail.",
        "",
        "`witness` names the pooled token whose value the (interpolated) "
        "quantile is nearest to, as `item@position`; `position` is 0-based "
        "among that item's scored answer positions, after any compare-time "
        "`--num-eval-tokens` cap. Ties resolve to the earliest item/position, "
        "so witnesses are reproducible across runs and `--jobs`.",
    ]
    for metric, block in pooled.items():
        heading, blurb = _POOLED_METRIC_HEADINGS.get(
            metric, (metric, f"per-token {metric}"))
        a_side, b_side = block["candidate_a"], block["candidate_b"]
        out += ["", f"### {heading}", "", f"{blurb}.", ""]
        headers = ["level", f"a = {a_label}", "witness (a)",
                   f"b = {b_label}", "witness (b)", "b − a"]
        rows = []
        tail = set(block.get("degradation_tail") or ())
        for la, lb, d in zip(a_side["levels"], b_side["levels"],
                             block["delta_b_minus_a"]):
            label = _POOLED_LADDER_LABELS.get(la["name"], la["name"]).strip()
            if la["name"] in tail:
                label = f"**{label}**"
            rows.append([label,
                         f"{la['value']:.6g}", _fmt_witness(la.get("witness")),
                         f"{lb['value']:.6g}", _fmt_witness(lb.get("witness")),
                         f"{d['delta_b_minus_a']:+.3g}"])
        out.append(_render_table(
            headers, rows,
            aligns=["left", "right", "left", "right", "left", "right"]))
        out += _format_critical_samples(block, a_label, b_label)
    return out


def _format_critical_samples(block, a_label: str, b_label: str) -> list[str]:
    """Who owns each far tail. A pooled quantile says how bad the tail is; it
    does not say where it came from, and in practice a handful of items
    usually own it."""
    crit = block.get("critical_samples")
    if not crit:
        return []
    metric = block["metric"]
    companion = crit.get("companion_metric")
    out = ["", f"**{metric} tail composition** — items holding tokens at or "
               f"beyond each candidate's own quantile "
               f"(top {_CRITICAL_SAMPLE_TABLE_ROWS} per side; the JSON holds "
               "every sample and every token position)."]
    headers = ["level", "candidate", "item", "tail tokens", "of item",
               f"item {'max' if crit['side'] == 'upper' else 'min'} {metric}"]
    if companion:
        headers.append(f"worst {companion} there")
    rows = []
    for lname, per_level in crit["levels"].items():
        label = _POOLED_LADDER_LABELS.get(lname, lname).strip()
        for side_key, side_label in (("candidate_a", a_label),
                                     ("candidate_b", b_label)):
            samples = per_level[side_key][:_CRITICAL_SAMPLE_TABLE_ROWS]
            if not samples:
                rows.append([label, side_label, "—", "—", "—", "—"]
                            + ([""] if companion else []))
                continue
            for smp in samples:
                row = [label, side_label, smp["item_key"],
                       str(smp["tail_token_count"]),
                       f"{100 * smp['tail_token_fraction']:.1f}%",
                       f"{smp['item_extreme']:.6g}"]
                if companion:
                    row.append(f"{smp['companion_min']:.6g}"
                               if "companion_min" in smp else "—")
                rows.append(row)
    aligns = ["left", "left", "left", "right", "right", "right"]
    if companion:
        aligns.append("right")
    out.append("")
    out.append(_render_table(headers, rows, aligns=aligns))
    return out


# --------------------------------------------------------------------------- #
