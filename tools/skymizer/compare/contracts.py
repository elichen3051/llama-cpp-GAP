# compare/contracts.py -- Constants, schema version, default metric/primary/CI/bucket policy, the
# AlignmentError/MissingMetricError exceptions, and min_bootstrap_iters.
# Sliced VERBATIM from paired_compare.py (R3 package split); bodies unchanged.
import math

# v3: significant paired verdicts say A/B closer (to the shared reference),
# and non-finite metrics abort instead of being removed from the paired sample.
#

# v2: the top-K pipeline was removed — `inputs.top_k` and
# `execution.args.kld_mode` are gone from the JSON (the markdown "Logits
# format" line is kind-aware now, but that is renderer-only). Numeric metric
# blocks are unchanged from v1.
SCHEMA_VERSION = "vlm-paired-compare-v3"

DEFAULT_METRICS = ("nll", "kld", "reversed_kld", "js_kld", "ear",
                   "ear_20", "ear_10", "ear_5",
                   "ear_20_normalized", "ear_10_normalized", "ear_5_normalized",
                   "ear_64", "ear_64_normalized",
                   "same_top_rate", "mse_dp")

# Multiplicity policy. Every base metric is reported item- AND token-weighted,
# so a complete v5 run emits 30 nominal 95% verdicts (26 for v4).
# In the original 14-cell grid, exchangeable A/B data (identical distributions,
# shared per-item latent, n=50) gave an uncorrected family-wise false-positive rate of 0.35.
#
# So exactly ONE cell is confirmatory: its interval is the report's claim and
# spends the whole alpha. Every other cell is EXPLORATORY and carries a
# Holm-Bonferroni-adjusted p-value across the rest of the family.
#
# The default primary is forward KLD, ITEM-weighted. Forward KLD is the
# quantity the tool exists to measure; the ITEM is the exchangeable unit --
# what was independently drawn when the dataset was built, and what the
# bootstrap actually resamples -- so the item-weighted mean is the estimand
# whose CI licenses inference to UNSEEN items from the same population. The
# ~1k tokens inside one answer share that item's image/topic/prefix: they are
# correlated evidence, not fresh observations. (Pre-registered as the primary
# in knowledge/quantization-eval-sop.md.)
#
# The token-weighted row is always reported: it is llama-perplexity's corpus
# aggregation (a VLM arm and a wikitext2 arm state one estimand), but as a
# confirmatory endpoint it is the weaker story -- a RATIO estimator whose
# weights are themselves random under resampling, where one 2,000-token
# answer outvotes twenty 100-token answers. --primary-weighting token
# switches which one is confirmatory.
DEFAULT_PRIMARY_METRIC = "kld"
DEFAULT_PRIMARY_WEIGHTING = "item"

# Per-token metric columns retained per item and pooled (flattened) across
# items for the descriptive distribution ladders. float32, ~4 KB per item per
# column at npos=1024 -- nothing like the vocab-wide buffers -- and kept
# because quantiles do not decompose into per-slice sums.
POOLED_TOKEN_METRICS = ("kld", "ear", "dp")

# The quantile ladder, in llama-perplexity --kl-divergence's print order and
# with its convention (linear interpolation between neighbouring order
# statistics = numpy's default = Hyndman-Fan type 7). q = 1.0 / 0.0 are the
# exact extremes.
POOLED_LADDER = (
    ("max", 1.0), ("p999", 0.999), ("p99", 0.99), ("p95", 0.95), ("p90", 0.90),
    ("median", 0.5),
    ("p10", 0.10), ("p05", 0.05), ("p01", 0.01), ("p001", 0.001), ("min", 0.0),
)

# Which end of each ladder is the DEGRADATION tail. KLD is lower-is-better, so
# its worst tokens are the top (p99/p999/max); EAR is higher-is-better
# (EAR = 1 - TV), so its worst tokens are the bottom (p01/p001/min) -- a
# literal "max EAR" sits at ~1.0 for any candidate worth measuring and says
# nothing.
# `dp` is SIGNED (p_cand - p_ref at the target, in pp), so both ends matter:
# a systematic over-confidence and a systematic under-confidence are different
# failures and the median says which way the mass moved. Nothing else the tool
# reports can see that -- mse_dp and rms_dp are unsigned by construction.
POOLED_TAIL_ROWS = {"kld": ("p99", "p999", "max"),
                    "ear": ("p01", "p001", "min"),
                    "dp": ("max", "p999", "median", "p001", "min")}


# "t" (default) is the classical paired Student-t interval on the per-item
# deltas (mean +- t_{n-1, 1-alpha/2} * s/sqrt(n); the token weighting uses
# the same ratio-linearized SE the studentized method does). It is the one
# method with NO bootstrap: deterministic, seed-free, and the SE it uses is
# the closed form Efron & Tibshirani (1986) cite as the case where
# resampling is unnecessary. On the right-skewed per-item deltas this tool
# produces its measured coverage is equal to or above the bootstrap-t's at
# n = 25..100 with a 3-33% narrower interval (docs/compare.md, "Interval
# construction"; review-functionality/measure_ci_coverage.py regenerates
# the table). The three bootstrap constructions are kept as options.
# Per-item TAIL statistics of a per-token column: each item's own p99 /
# p99.9 / maximum of that column (the same interpolation convention as the
# pooled ladder, so the per-item p99 and the pooled p99 are one definition
# at two scopes), each with the answer position it occurred at. Per item
# they are inspection aids -- "which token blew up in which answer"; across
# items each level is a legitimate per-item scalar and gets the paired test
# every other per-item metric gets, in its OWN exploratory Holm family. They
# are never the confirmatory endpoint: a per-item maximum is a single-token
# statistic and far heavier-tailed than a mean.
PER_ITEM_TAIL_METRICS = ("kld",)
PER_ITEM_TAIL_LADDER = (("p99", 0.99), ("p999", 0.999), ("max", 1.0))

# Per-token INTEGER annotation columns carried beside POOLED_TOKEN_METRICS in
# the token dicts (same alignment: one value per scored position). "target"
# is the teacher-forced target token id, so a tail witness can name the
# token, not just its position. Never a metric: no ladder, no test.
TOKEN_ANNOTATION_COLUMNS = ("target",)

CI_METHODS = ("t", "studentized", "bca", "percentile")
DEFAULT_CI_METHOD = "t"
BOOTSTRAP_CI_METHODS = tuple(m for m in CI_METHODS if m != "t")


def min_bootstrap_iters(confidence_level: float,
                        ci_method: str = DEFAULT_CI_METHOD) -> int:
    """Smallest --bootstrap-iters that can carry a CI at this level; 0 for
    the t interval, which draws no replicates at all.

    The percentile endpoints are the empirical alpha/2 and 1-alpha/2
    quantiles of the replicate sample, so they are only meaningful when a
    decent number of replicates fall strictly BEYOND each endpoint. We
    require at least 10 (hence ceil(10 / (alpha/2))) and never fewer than
    200 replicates overall. Below that the "interval" degenerates towards
    the min/max of the replicates: at --bootstrap-iters 1 lower == upper,
    which excludes 0 for every metric and prints a significant verdict for
    all of them.

    The two corrected methods double the floor. BCa's bias/skew correction
    routinely pushes the requested levels FURTHER into the tails than
    alpha/2 (a typical right-skewed per-item delta at n=25 lands near
    0.06 / 0.994); the studentized interval reads the same alpha/2 levels
    but of a t* distribution with much heavier tails, so those quantiles
    are noisier. Both want more replicates out there than percentile does.

    At the 0.95 default this is 400 (percentile) / 800 (bca, studentized);
    at 0.99, 2000 / 4000. It is a floor, not a recommendation -- 2000+ is
    the usual working number and the CLI warns below it.
    """
    if ci_method == "t":
        return 0
    alpha = 1.0 - confidence_level
    floor = max(200, math.ceil(10.0 / (alpha / 2.0)))
    return floor * 2 if ci_method in ("bca", "studentized") else floor


BOOTSTRAP_ITERS_ADVISORY = 2000   # below this the CLI warns (MC noise), not fails


LOWER_IS_BETTER = {
    "nll", "kld", "reversed_kld", "js_kld", "mse_dp", "rms_dp", "ppl_ratio", "ppl",
}


class AlignmentError(ValueError):
    """Raised when the three dirs do not describe the same evaluated items."""


class NonFiniteMetricError(ValueError):
    """Raised when a metric is NaN or infinite; paired inference is invalid."""


class MissingMetricError(ValueError):
    """Raised when an explicitly requested metric is unavailable."""


# --------------------------------------------------------------------------- #
