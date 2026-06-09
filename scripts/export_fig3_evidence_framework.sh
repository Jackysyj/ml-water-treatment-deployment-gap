#!/usr/bin/env bash
# Export Fig. 3 from its draw.io source.
#
# The active Fig. 3 source is figures/fig3_evidence_framework.drawio.
# In the full project tree, the local headless draw.io exporter is expected at
# ../tools/drawio/export.sh relative to this project root. For a copied package,
# set DRAWIO_EXPORTER to another compatible draw.io export wrapper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPORTER="${DRAWIO_EXPORTER:-$PROJECT_ROOT/../tools/drawio/export.sh}"
SCALE="${1:-2}"

if [[ -f "$PROJECT_ROOT/figures/fig3_evidence_framework.drawio" ]]; then
  DRAWIO_SOURCE="$PROJECT_ROOT/figures/fig3_evidence_framework.drawio"
  PNG_OUTPUT="$PROJECT_ROOT/figures/fig3_evidence_framework.png"
elif [[ -f "$PROJECT_ROOT/derived_outputs/fig3_evidence_framework.drawio" ]]; then
  DRAWIO_SOURCE="$PROJECT_ROOT/derived_outputs/fig3_evidence_framework.drawio"
  PNG_OUTPUT="$PROJECT_ROOT/derived_outputs/fig3_evidence_framework.png"
else
  echo "ERROR: Fig. 3 draw.io source not found." >&2
  echo "Expected either:" >&2
  echo "  $PROJECT_ROOT/figures/fig3_evidence_framework.drawio" >&2
  echo "  $PROJECT_ROOT/derived_outputs/fig3_evidence_framework.drawio" >&2
  exit 1
fi

mkdir -p "$(dirname "$PNG_OUTPUT")"

if [[ ! -x "$EXPORTER" ]]; then
  echo "ERROR: draw.io exporter not found or not executable: $EXPORTER" >&2
  echo "Set DRAWIO_EXPORTER=/path/to/export.sh or use draw.io Desktop export manually." >&2
  exit 1
fi

"$EXPORTER" "$DRAWIO_SOURCE" "$PNG_OUTPUT" "$SCALE"
