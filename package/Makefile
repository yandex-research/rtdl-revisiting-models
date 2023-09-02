.PHONY: default clean coverage docs docserver doctest format lint pages pre-commit typecheck

PACKAGE_ROOT = paper_tabular_dl_revisiting_models.py
DOCS_DIR = ../docs

default:
	echo "Hello, World!"

clean:
	find . -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -f .coverage
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist

docs:
	rm $(DOCS_DIR)/*.html
	rm $(DOCS_DIR)/*.js
	pdoc $(PACKAGE_ROOT) --docformat google --template-directory $(DOCS_DIR) -o $(DOCS_DIR)

docserver:
	pdoc $(PACKAGE_ROOT) --docformat google --template-directory $(DOCS_DIR)

doctest:
	xdoctest $(PACKAGE_ROOT)

lint:
	black $(PACKAGE_ROOT) --check
	ruff check .

pre-commit: clean lint doctest test docs typecheck

test:
	# PYTHONPATH='.' pytest $(PACKAGE_ROOT) $(ARGV)
	PYTHONPATH='.' pytest tests $(ARGV)

typecheck:
	mypy $(PACKAGE_ROOT)
