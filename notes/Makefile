# Compile the editable manual sources and publish the checked PDF in docs/.
# The manual target checks PDF metadata and unresolved LaTeX references.
# The render target writes page PNGs under notes/tmp/pdfs for visual inspection.

MANUAL := venus_manual
MANUAL_PDF := ../docs/$(MANUAL).pdf
LATEXMK ?= latexmk
PDFINFO ?= pdfinfo
PDFTOPPM ?= pdftoppm
PYTHON ?= python3

.PHONY: manual render clean

# Stop on a TeX error, then check the resulting PDF and converged reference log.
manual:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MANUAL).tex
	$(PDFINFO) $(MANUAL).pdf
	$(PYTHON) ../scripts/check_manual_log.py $(MANUAL).log
	mkdir -p ../docs
	mv -f $(MANUAL).pdf $(MANUAL_PDF)

# Render only after the manual target has produced a checked PDF.
render: manual
	mkdir -p tmp/pdfs
	$(PDFTOPPM) -png -r 96 $(MANUAL_PDF) tmp/pdfs/$(MANUAL)

# Remove intermediate TeX products while retaining the published PDF and sources.
clean:
	$(LATEXMK) -C $(MANUAL).tex
	rm -f $(MANUAL).bbl $(MANUAL).run.xml
