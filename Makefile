NB = $(sort $(wildcard notebooks/*.ipynb))
.PHONY: help clean lint test doc dist release

help:
	@echo "clean    remove non-source files and clean source files"
	@echo "lint     check style"
	@echo "test     run tests and check coverage"
	@echo "doc      generate HTML documentation and check links"
	@echo "dist     package (source & wheel)"
	@echo "release  package and upload to PyPI"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ doc/_build/
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)

lint:
	ruff check

# Matplotlib doesn't print to screen. Also faster.
export MPLBACKEND = agg
# Virtual framebuffer nonetheless needed for the pyqtgraph backend.
export DISPLAY = :99

test:
	pytest

clean-doc:
	rm -rf doc/_build

doc:
	sphinx-build -b html -d doc/_build/doctrees doc doc/_build/html
	sphinx-build -b linkcheck -d doc/_build/doctrees doc doc/_build/linkcheck

dist: clean
	python -m build
	ls -lh dist/*
	twine check dist/*

release: dist
	twine upload dist/*