#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf main.tex
else
  if ! command -v pdflatex >/dev/null 2>&1; then
    echo "error: pdflatex not found. Install MacTeX or BasicTeX first." >&2
    exit 1
  fi
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
fi

echo "Built $ROOT_DIR/main.pdf"
