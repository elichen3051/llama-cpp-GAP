#!/usr/bin/env bash
# One (reference model, prepared corpus) segment: every candidate in order, each collected and verified.
# Layout: RUNS_ROOT/our-llm-kld-records/MODEL/CORPUS/candidates/LABEL/{llm.log,llm-kld/,verify.log,verify-checks.csv}
#
# usage: run_segment.sh REF_MODEL PREPARED_DIR RUNS_ROOT MODEL CORPUS LABEL=CAND_MODEL [LABEL=CAND_MODEL ...]
#   MODEL / CORPUS                   single path components, e.g. qwen3.5-4b / pg-full-rss-article (CORPUS must be the
#                                    corpus_name of PREPARED_DIR/corpus_articles.json)
#   LABEL                            candidate--<provider>--<quant>
#   env ALLOW_VOCAB_ATTR_MISMATCH=1  pass --allow-vocab-attr-mismatch (automatic for the Gemma family)
# Parity anchor: the first candidate of the list whose verify-checks.csv passes, else the first one collected in this run.
# Every newly collected candidate is verified against it; every already verified candidate is re-checked against it once
# (CAND_DIR/parity-<anchor label>-<first 12 hex of sha256 of the anchor's llm-kld/collect_meta.json>.csv, so a re-collected
# anchor of the same label never reuses an old receipt). Resume: verified candidates are not collected again. Stops before any
# collection: a repeated LABEL, an existing candidate without a passing verify-checks.csv, a verified candidate whose
# llm-kld/corpus_articles.json is missing or differs from PREPARED_DIR's, a missing anchor collect_meta.json, a failing parity
# receipt or re-check.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"
usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 6 ]] || usage
REF=$1; PREPARED=$2; ROOT=$3; MODEL=$4; CORPUS=$5; shift 5
component() { [[ -n "$1" && "$1" != . && "$1" != .. && "$1" == "${1//\//}" ]]; }
component "$MODEL" && component "$CORPUS" || die "MODEL and CORPUS must be single path components: $MODEL / $CORPUS"
UBATCH=$(ubatch_for "$MODEL") || die "unknown model: $MODEL"
LABEL_RE='^candidate--[A-Za-z0-9._]+(-[A-Za-z0-9._]+)*--[A-Za-z0-9._][A-Za-z0-9._-]*$'
SEEN=$'\n'
for spec in "$@"; do
    [[ "$spec" == *=* ]] || die "candidate must be LABEL=PATH: $spec"
    [[ "${spec%%=*}" =~ $LABEL_RE ]] || die "LABEL must be candidate--<provider>--<quant>: ${spec%%=*}"
    [[ "$SEEN" != *$'\n'"${spec%%=*}"$'\n'* ]] || die "LABEL appears more than once: ${spec%%=*}"
    SEEN+="${spec%%=*}"$'\n'
done
require_fork
# Absolute before use: the parity re-check below runs after `cd FORK_REPO`.
PREPARED=$(abspath "$PREPARED"); ROOT=$(abspath "$ROOT")
WAIVER=()
[[ "${ALLOW_VOCAB_ATTR_MISMATCH:-0}" != 1 ]] || WAIVER=(--allow-vocab-attr-mismatch)

# The prepared corpus must be the one this segment is named after.
ARTICLES=$PREPARED/corpus_articles.json
if [[ -f "$ARTICLES" ]]; then
    "$PYTHON" - "$ARTICLES" "$CORPUS" <<'PY' || die "CORPUS $CORPUS is not the corpus_name of $ARTICLES"
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    sys.exit(0 if json.load(stream)["protocol"]["corpus_name"] == sys.argv[2] else 1)
PY
else
    _missing "missing file: $ARTICLES (cannot check CORPUS against its corpus_name)"
fi

SEG=$ROOT/our-llm-kld-records/$MODEL/$CORPUS/candidates

# Pre-scan, before any re-check or collection: the anchor is the first verified candidate in list order; every verified
# candidate must hold the same corpus_articles.json as PREPARED_DIR; no existing candidate may be unverified.
ANCHOR=
for spec in "$@"; do
    CDIR=$SEG/${spec%%=*}
    if checks_pass "$CDIR/verify-checks.csv"; then
        ANCHOR=${ANCHOR:-$CDIR}
        if [[ -f "$ARTICLES" ]]; then
            COPY=$CDIR/llm-kld/corpus_articles.json
            [[ -f "$COPY" ]] || die "verified candidate has no corpus copy (missing file): $COPY"
            cmp -s -- "$COPY" "$ARTICLES" \
                || die "verified candidate was collected on another preparation of the corpus: $COPY differs from $ARTICLES"
        fi
    elif [[ -e "$CDIR" ]]; then
        die "existing candidate without a passing verify-checks.csv, refusing to continue (move it aside): $CDIR"
    fi
done
if [[ -n "$ANCHOR" ]]; then
    # Receipts are named after the anchor collection: its label and a fingerprint of its collect_meta.json.
    META=$ANCHOR/llm-kld/collect_meta.json
    if [[ -f "$META" ]]; then
        FINGERPRINT=$("$PYTHON" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()[:12])' "$META")
    else
        _missing "missing file: $META (the anchor's parity receipts are named after it)"
        FINGERPRINT=missing
    fi
    PARITY_NAME=parity-$(basename -- "$ANCHOR")-$FINGERPRINT
    log "parity anchor (first verified candidate): $ANCHOR ($FINGERPRINT)"
    # Re-check every other verified candidate against the anchor before any GPU work.
    for spec in "$@"; do
        CDIR=$SEG/${spec%%=*}
        [[ "$CDIR" != "$ANCHOR" ]] && checks_pass "$CDIR/verify-checks.csv" || continue
        CSV=$CDIR/$PARITY_NAME.csv
        if [[ -e "$CSV" ]]; then
            checks_pass "$CSV" || die "parity re-check against the anchor failed earlier: $CSV"
            log "parity against the anchor already verified: $CSV"
            continue
        fi
        ( cd -- "$FORK_REPO" && run_logged "$CDIR/$PARITY_NAME.log" "$PYTHON" "$FORK_REPO/tools/gap/cli/verify_article_collection.py" \
            --articles "$CDIR/llm-kld/corpus_articles.json" --collection "$CDIR/llm-kld" "${RUNTIME_ARGS[@]}" --n-ubatch "$UBATCH" \
            --parity-with "$ANCHOR/llm-kld" --checks-out "$CSV" ) \
            || die "parity re-check against the anchor failed: $CSV"
    done
fi

for spec in "$@"; do
    LABEL=${spec%%=*}; CAND=${spec#*=}; CDIR=$SEG/$LABEL
    if checks_pass "$CDIR/verify-checks.csv"; then
        log "already verified, skipping: $CDIR"
        continue
    fi
    PARITY=()
    [[ -z "$ANCHOR" ]] || PARITY=(--parity-with "$ANCHOR")
    "$HERE/llm_kld.sh" "$MODEL" "$REF" "$CAND" "$PREPARED" "$CDIR" ${WAIVER[@]+"${WAIVER[@]}"} ${PARITY[@]+"${PARITY[@]}"}
    ANCHOR=${ANCHOR:-$CDIR}
done
log "segment complete: $SEG"
