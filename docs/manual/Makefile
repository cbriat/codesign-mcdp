TEX  = codesign-mcdp-manual.tex
BASE = codesign-mcdp-manual
PDF  = $(BASE).pdf
BIB  = references.bib

LATEX  = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

.PHONY: all clean distclean

all: $(PDF)

# Full sequence. The first pass writes the .aux with the \citation
# records, bibtex turns those into the .bbl, and the further passes pull
# the bibliography in and settle every cross reference, page number, and
# table-of-contents entry. Three passes after bibtex are needed because
# adding the bibliography shifts the page numbers that the contents and
# the back references point at.
$(PDF): $(TEX) $(BIB)
	$(LATEX) $(BASE)
	$(BIBTEX) $(BASE)
	$(LATEX) $(BASE)
	$(LATEX) $(BASE)
	$(LATEX) $(BASE)

clean:
	rm -f *.aux *.log *.toc *.out *.fls *.fdb_latexmk *.synctex.gz \
	      *.bbl *.blg

distclean: clean
	rm -f $(PDF)
