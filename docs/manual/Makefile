TEX = codesign-mcdp-manual.tex
PDF = codesign-mcdp-manual.pdf

.PHONY: all clean distclean

all: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.log *.toc *.out *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDF)
