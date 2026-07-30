#!/bin/bash
# Kompiliert nur den bestehenden Build-Ordner neu (kein Neu-Fuellen der Platzhalter).
# Fuer schnelles Iterieren an report/build/L<n>/*.tex nach create_report.py.
# Aufruf: ./rebuild_pdf.sh 1
set -e

LINAC="${1:?Nutzung: ./rebuild_pdf.sh <linac_id>}"
BUILD_DIR="report/build/L${LINAC}"

if [ ! -f "${BUILD_DIR}/main.tex" ]; then
    echo "[X] ${BUILD_DIR}/main.tex nicht gefunden. Erst 'python create_report.py ${LINAC}' ausfuehren."
    exit 1
fi

cd "${BUILD_DIR}"
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
echo "[✓] PDF aktualisiert: ${BUILD_DIR}/main.pdf"
