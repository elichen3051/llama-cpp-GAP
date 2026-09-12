#!/usr/bin/env bash
# Download and verify the two frozen corpora behind runs/ (stage 0, before prepare_corpus.sh).
#
# usage: get_corpora.sh [--out DIR] [--pg-limit N] [--expect-manifest FILE] [wikitext|pg|all|verify]
#   verify                  download nothing; check the files already under --out (e.g. the frozen PG copy placed
#                           there by hand) against the identities below
#   --out DIR               output root (default: ./corpora)
#   --pg-limit N            convert only the first N essays of the feed (mechanics check; verification skipped)
#   --expect-manifest FILE  frozen PG manifest.json (per-article byte spans / sha256) to compare article by article
#   exit status 0 only when every requested corpus verifies against the frozen identity below
#
# WikiText-2 test split (reproducible; verified 2026-09-12 from both `main` and the pinned snapshot):
#   zip   https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip
#         sha256 ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11   4721645 bytes
#   file  wikitext-2/wikitext-2-raw/wiki.test.raw
#         sha256 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08   1290590 bytes
#   index wikitext-2/articles.json (60 explicit articles; copied from ../campaign/bundle/articles.json)
#         sha256 63216bfa0d687101963fada39c8b7434e9953eef87f8fdb0e5207371c0bf6b72
#   Same archive as llama.cpp's scripts/get-wikitext-2.sh, pinned to the dataset commit the campaign used.
#
# PG = the 217 Paul Graham essays listed by http://www.aaronsw.com/2002/feeds/pgessays.rss (best effort):
#   the campaign's frozen copy (2026-09-06) is llama.cpp's scripts/get-pg.sh with n=217, except that every page was
#   first normalized with Python html5lib 1.1 before the C++ html2text 2.4.0 (Debian/Ubuntu package), then
#   `tail -n +4 | sed -E 's/^[[:space:]]+//g' | fmt -w 80` with GNU coreutils under LC_ALL=C.UTF-8:
#   file  pg-normalized-html5lib1.1-html2text2.4.0/pg.txt
#         sha256 26db5717f58a11a8ed9c24dab9acffc557bcb9d7697b733f44039f13cca4e082   3179044 bytes
#   index pg-normalized-html5lib1.1-html2text2.4.0/manifest.json (byte spans per essay)
#         frozen copy: sha256 0abeb0d4ce01100c4a8781599ee932daf0e80c85aa20b7286168c796b9da198f   373445 bytes
#   The feed is live and the converter stack matters (macOS/BSD fmt differs from GNU fmt). A SHA256 mismatch means
#   the frozen pg.txt and manifest.json must be taken from the original bundle instead of this download.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" >&2; exit 2; }

WIKI_ZIP_URL=https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip
WIKI_ZIP_SHA=ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11
WIKI_SHA=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08; WIKI_BYTES=1290590
INDEX_SHA=63216bfa0d687101963fada39c8b7434e9953eef87f8fdb0e5207371c0bf6b72
PG_FEED=http://www.aaronsw.com/2002/feeds/pgessays.rss
PG_SHA=26db5717f58a11a8ed9c24dab9acffc557bcb9d7697b733f44039f13cca4e082; PG_BYTES=3179044; PG_COUNT=217
PG_MANIFEST_SHA=0abeb0d4ce01100c4a8781599ee932daf0e80c85aa20b7286168c796b9da198f; PG_MANIFEST_BYTES=373445

OUT=corpora; LIMIT=; EXPECT=; WHAT=all
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT=$2; shift 2 ;;
        --pg-limit) LIMIT=$2; shift 2 ;;
        --expect-manifest) EXPECT=$(abspath "$2"); shift 2 ;;
        wikitext|pg|all|verify) WHAT=$1; shift ;;
        *) usage ;;
    esac
done
OUT=$(abspath "$OUT"); mkdir -p -- "$OUT"
sha_of() { sha256_file "$1" | cut -d' ' -f1; }
check() { # check LABEL FILE SHA [BYTES]
    local got; got=$(sha_of "$2")
    if [[ "$got" == "$3" && ( -z "${4:-}" || "$(file_size "$2")" == "$4" ) ]]; then log "verified  $1: $2"; return 0; fi
    log "MISMATCH  $1: $2 sha256=$got bytes=$(file_size "$2") expected $3${4:+ $4 bytes}"; return 1
}
STATUS=0

if [[ "$WHAT" == verify ]]; then
    P=$OUT/pg-normalized-html5lib1.1-html2text2.4.0
    first_existing() { for f in "$@"; do [[ -f "$f" ]] && { echo "$f"; return; }; done; echo "$1"; }
    verify_present() { # LABEL SHA BYTES REQUIRED(1/0) FILE...
        local label=$1 sha=$2 bytes=$3 required=$4; shift 4; local f; f=$(first_existing "$@")
        if [[ -f "$f" ]]; then check "$label" "$f" "$sha" "$bytes" || STATUS=1
        elif [[ "$required" == 1 ]]; then log "missing   $label: $1"; STATUS=1
        else log "not shipped (optional)  $label"; fi
    }
    verify_present "wiki.test.raw" "$WIKI_SHA" "$WIKI_BYTES" 1 "$OUT/wiki.test.raw" "$OUT/wikitext-2/wikitext-2-raw/wiki.test.raw"
    verify_present "pg.txt (217 essays)" "$PG_SHA" "$PG_BYTES" 1 "$OUT/pg.txt" "$P/pg.txt"
    verify_present "wikitext-2 zip" "$WIKI_ZIP_SHA" 4721645 0 "$OUT/wikitext-2/wikitext-2-raw-v1.zip" "$OUT/wikitext-2-raw-v1.zip"
    verify_present "articles.json (60-article index)" "$INDEX_SHA" 10095 0 "$OUT/wikitext-2/articles.json" "$OUT/articles.json"
    verify_present "pg manifest.json" "$PG_MANIFEST_SHA" "$PG_MANIFEST_BYTES" 0 "$P/manifest.json" "$OUT/manifest.json"
fi

if [[ "$WHAT" == wikitext || "$WHAT" == all ]]; then
    command -v curl >/dev/null && command -v unzip >/dev/null || die "wikitext needs curl and unzip"
    W=$OUT/wikitext-2; mkdir -p -- "$W"
    log "downloading $WIKI_ZIP_URL"
    curl -fsSL --retry 3 -o "$W/wikitext-2-raw-v1.zip" "$WIKI_ZIP_URL"
    check "wikitext-2 zip" "$W/wikitext-2-raw-v1.zip" "$WIKI_ZIP_SHA" 4721645 || STATUS=1
    (cd "$W" && unzip -q -o wikitext-2-raw-v1.zip wikitext-2-raw/wiki.test.raw)
    check "wiki.test.raw" "$W/wikitext-2-raw/wiki.test.raw" "$WIKI_SHA" "$WIKI_BYTES" || STATUS=1
    SRC_INDEX=$(dirname -- "${BASH_SOURCE[0]}")/../campaign/bundle/articles.json
    if [[ -f "$SRC_INDEX" ]]; then
        cp -p -- "$SRC_INDEX" "$W/articles.json"
        check "articles.json (60-article index)" "$W/articles.json" "$INDEX_SHA" || STATUS=1
    else
        log "note: ../campaign/bundle/articles.json not found; prepare_corpus.sh needs the 60-article index"
    fi
fi

if [[ "$WHAT" == pg || "$WHAT" == all ]]; then
    for c in curl html2text tail sed fmt; do command -v "$c" >/dev/null || die "pg needs $c (C++ html2text 2.4.0, GNU coreutils)"; done
    fmt --version 2>/dev/null | grep -q GNU || log "warning: fmt is not GNU coreutils; the frozen corpus used GNU fmt -w 80"
    "$PYTHON" -c 'import html5lib; assert html5lib.__version__ == "1.1", html5lib.__version__' 2>/dev/null \
        || log "warning: Python html5lib 1.1 not importable via $PYTHON; the frozen corpus used html5lib 1.1"
    export LC_ALL=C.UTF-8
    P=$OUT/pg-normalized-html5lib1.1-html2text2.4.0; mkdir -p -- "$P/essays"
    log "fetching feed $PG_FEED"
    curl -fsSL --retry 3 -o "$P/pgessays.rss" "$PG_FEED"
    grep html "$P/pgessays.rss" | sed -e 's/.*http/http/' -e 's/html.*/html/' > "$P/urls.txt"
    [[ -n "$LIMIT" ]] && { head -n "$LIMIT" "$P/urls.txt" > "$P/urls.limited" && mv "$P/urls.limited" "$P/urls.txt"; }
    N=$(wc -l < "$P/urls.txt" | tr -d ' ')
    log "$N essay urls (frozen corpus: $PG_COUNT)"
    : > "$P/pg.txt"; : > "$P/spans.tsv"
    c=1
    while read -r url; do
        cc=$(printf '%03d' "$c"); one=$P/essays/pg-$cc-one.txt
        log "[$cc/$N] $url"
        curl -fsSL --retry 3 -o "$P/essays/pg-$cc.html" "$url"
        "$PYTHON" - "$P/essays/pg-$cc.html" <<'PY' > "$P/essays/pg-$cc.normalized.html"
import sys, html5lib
raw = open(sys.argv[1], "rb").read()
sys.stdout.write(html5lib.serialize(html5lib.parse(raw), tree="etree"))
PY
        html2text < "$P/essays/pg-$cc.normalized.html" | tail -n +4 | sed -E 's/^[[:space:]]+//g' | fmt -w 80 > "$one"
        start=$(file_size "$P/pg.txt"); cat "$one" >> "$P/pg.txt"; end=$(file_size "$P/pg.txt")
        printf '%s\t%s\t%s\t%s\t%s\n' "$cc" "$url" "$start" "$end" "$(sha_of "$one")" >> "$P/spans.tsv"
        c=$((c + 1)); sleep 1
    done < "$P/urls.txt"
    "$PYTHON" - "$P" "$PG_FEED" <<'PY'
import csv, datetime, json, sys
root, feed = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(f"{root}/spans.tsv"), delimiter="\t"))
articles = [{"id": r[0], "url": r[1], "byte_start": int(r[2]), "byte_end_exclusive": int(r[3]), "sha256": r[4]} for r in rows]
json.dump({"source_feed": feed, "retrieved": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "pipeline": "html5lib 1.1 serialize -> html2text 2.4.0 -> tail -n +4 -> sed -E 's/^[[:space:]]+//g' -> fmt -w 80 (LC_ALL=C.UTF-8)",
           "articles": articles}, open(f"{root}/manifest.json", "w"), indent=1)
PY
    if [[ -n "$LIMIT" ]]; then
        log "pg: --pg-limit given, mechanics only; frozen identity not checked"
    else
        check "pg.txt ($N essays)" "$P/pg.txt" "$PG_SHA" "$PG_BYTES" || STATUS=1
        if [[ -n "$EXPECT" ]]; then
            "$PYTHON" - "$P/manifest.json" "$EXPECT" <<'PY' || STATUS=1
import json, sys
mine = json.load(open(sys.argv[1]))["articles"]; ref = json.load(open(sys.argv[2]))
ref = ref.get("articles") or ref.get("items") or ref
bad = 0
for i, (a, b) in enumerate(zip(mine, ref)):
    same = a["byte_start"] == b.get("byte_start") and a["byte_end_exclusive"] == b.get("byte_end_exclusive") and \
           (b.get("sha256") in (None, a["sha256"]))
    if not same:
        bad += 1
        if bad <= 5: print(f"article {i:03d} differs: mine {a['byte_start']}..{a['byte_end_exclusive']} {a['sha256'][:12]} / frozen {b.get('byte_start')}..{b.get('byte_end_exclusive')} {str(b.get('sha256'))[:12]}")
print(f"per-article comparison: {len(mine)} mine vs {len(ref)} frozen, {bad} differ")
sys.exit(1 if bad or len(mine) != len(ref) else 0)
PY
        fi
    fi
fi
[[ $STATUS -eq 0 ]] && log "all requested corpora verified" || log "VERIFICATION FAILED: use the frozen files from the original bundle for any corpus above"
exit $STATUS
