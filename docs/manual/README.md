# Manual

Source for the user manual.

## Files

- `codesign-mcdp-manual.tex` – LaTeX source.
- `codesign-mcdp-manual.pdf` – pre-built PDF (committed for convenience).

## Rebuilding

You need a working LaTeX distribution (TeX Live or MiKTeX). On
Debian/Ubuntu:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

The manual has no bibliography database (references are a plain numbered
list), so no `bibtex`/`biber` pass is required — but it does need two
LaTeX passes so the table of contents and the cross-references resolve.

From this directory, either use the Makefile:

```bash
make            # runs pdflatex twice
```

or, if you have `latexmk`, let it manage the passes automatically:

```bash
latexmk -pdf codesign-mcdp-manual.tex
```

or run `pdflatex` by hand:

```bash
pdflatex codesign-mcdp-manual.tex
pdflatex codesign-mcdp-manual.tex   # second pass populates TOC and refs
```

The document builds cleanly with `pdflatex` (TeX Live 2025). A handful of
`Overfull \hbox` warnings from long code-listing lines and bibliography
entries are cosmetic and can be ignored.

## Cleanup

`make clean` removes intermediate `.aux`, `.log`, `.toc`, `.out` and
`latexmk` bookkeeping files. `make distclean` also removes the `.pdf`.
With `latexmk`, `latexmk -c` cleans auxiliary files and `latexmk -C`
also removes the `.pdf`.
