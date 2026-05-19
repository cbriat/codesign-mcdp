# Manual

Source for the user manual.

## Files

- `codesign-mcdp-manual.tex` – LaTeX source.
- `codesign-mcdp-manual.pdf` – pre-built PDF (committed for convenience).

## Rebuilding

You need a working LaTeX distribution. On Debian/Ubuntu:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

Then from this directory:

```bash
make
# or, directly:
pdflatex codesign-mcdp-manual.tex
pdflatex codesign-mcdp-manual.tex   # second pass populates the TOC
```

## Cleanup

`make clean` removes intermediate `.aux`, `.log`, `.toc`, `.out` files.
`make distclean` also removes the `.pdf`.
