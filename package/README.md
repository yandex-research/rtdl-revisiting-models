# Python package

[Documentation](https://yandex-research.github.io/tabular-dl-revisiting-models)

# Development

**WARNING**: if you copy this project to create a similar package for your paper,
please, note that there are many places where you have to change names, links, etc.
There is no other way not to miss them except for reading all the files carefully.

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
