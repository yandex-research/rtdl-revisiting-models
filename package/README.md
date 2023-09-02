# Python package

[Documentation](https://yandex-research.github.io/tabular-dl-revisiting-models)

# Development

Set up the environment (replace `micromamba` with `conda` or `mamba` if needed):
```
micromamba create -f environment-dev.yaml
```

**Run linters and tests before committing:**
```
make pre-commit
```

Update documentation:
```
make docs
```

Publish the package to PyPI (requires PyPI account & configuration):
```
flit publish
```
