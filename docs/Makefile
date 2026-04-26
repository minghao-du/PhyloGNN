SPHINXOPTS ?=
SOURCEDIR = source
BUILDDIR = _build
PYTHON ?= python

.PHONY: help html doctest clean

help:
	$(PYTHON) -m sphinx -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	$(PYTHON) -m sphinx -b html -n -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

doctest:
	$(PYTHON) -m sphinx -b doctest -W "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"
