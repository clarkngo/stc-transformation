#!/usr/bin/env bash
# Offline smoke test for all four AI 420 HOS — no Gemini key used, no quota spent.
#
#   bash run_all.sh            # answer keys (does each unit work?)
#   bash run_all.sh starter    # starter code (is each unit ready for students?)
#
# Real calls go to the free services (Wikipedia, Open-Meteo, a web page, SQLite, Chroma);
# the Gemini model is replaced by a scripted stand-in (fakellm.py). Each HOS folder is
# copied to a temp directory first, so nothing is written into the repo.
set -u
VARIANT="${1:-answer}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../$([ "$VARIANT" = starter ] && echo starter-code || echo answer-key)"
TMP="$(mktemp -d)"
export PYTHONWARNINGS=ignore
fails=0
for n in 1 2 3 4; do
  dir=$(cd "$SRC" && ls -d hos-0$n-*)
  cp -R "$SRC/$dir" "$TMP/"
  echo; echo "################ HOS $n ($VARIANT): $dir"
  (cd "$TMP/$dir" && python "$HERE/smoke_hos$n.py" "$VARIANT" 2>&1 | grep -v 'EXPERIMENTAL\|check_feature_enabled') || fails=$((fails+1))
done
rm -rf "$TMP"
echo
if [ $fails -eq 0 ]; then echo "ALL FOUR HOS PASS ($VARIANT)"; else echo "$fails HOS HAD FAILURES ($VARIANT) — see FAIL lines above"; exit 1; fi
