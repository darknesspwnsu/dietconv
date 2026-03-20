#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
else
  if ! command -v pdflatex >/dev/null 2>&1; then
    cat >&2 <<'EOF'
error: pdflatex not found.

Install a TeX distribution, then rerun this script.

Recommended on macOS:
  brew install --cask basictex
  export PATH="/Library/TeX/texbin:$PATH"
  sudo tlmgr update --self
  sudo tlmgr install latexmk collection-latexrecommended collection-fontsrecommended natbib booktabs multirow geometry hyperref

User-space alternative that does not require the macOS installer:
  brew install tectonic

Larger alternative:
  brew install --cask mactex-no-gui
EOF
    exit 1
  fi
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
fi

echo "Built $ROOT_DIR/main.pdf"
