# Manual

Source for the user manual.

## Files

- `codesign-mcdp-manual.tex` – LaTeX source.
- `references.bib` – bibliography database (natbib/BibTeX).
- `codesign-mcdp-manual.pdf` – pre-built PDF (committed for convenience).

## Rebuilding

You need a working LaTeX distribution (TeX Live or MiKTeX). On
Debian/Ubuntu:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

The manual uses `natbib` with BibTeX over `references.bib`, so the build
needs a `bibtex` pass between LaTeX passes, plus further LaTeX passes so
the bibliography, the table of contents, and the cross-references all
resolve. Everything used is part of a plain TeX Live install; no extra
configuration is required.

From this directory, use the Makefile, which runs the whole sequence:

```bash
make            # pdflatex, bibtex, then pdflatex three more times
```

or, if you have `latexmk`, let it manage the passes automatically:

```bash
latexmk -pdf codesign-mcdp-manual.tex
```

or run the tools by hand:

```bash
pdflatex codesign-mcdp-manual.tex
bibtex   codesign-mcdp-manual
pdflatex codesign-mcdp-manual.tex
pdflatex codesign-mcdp-manual.tex
pdflatex codesign-mcdp-manual.tex
```

The document builds cleanly with `pdflatex` (TeX Live 2025): no errors, no
undefined citations or references. A handful of `Overfull \hbox` warnings
from long code-listing lines and bibliography entries are cosmetic and can
be ignored.

## Cleanup

`make clean` removes intermediate `.aux`, `.log`, `.toc`, `.out`, `.bbl`,
`.blg` and `latexmk` bookkeeping files. `make distclean` also removes the
`.pdf`.
With `latexmk`, `latexmk -c` cleans auxiliary files and `latexmk -C`
also removes the `.pdf`.
