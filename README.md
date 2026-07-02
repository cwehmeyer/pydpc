# pydpc - a Python package for Density Peak-based Clustering

![CI](https://github.com/cwehmeyer/pydpc/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/cwehmeyer/pydpc/branch/master/graph/badge.svg)](https://codecov.io/gh/cwehmeyer/pydpc)
[![CodeQL](https://github.com/cwehmeyer/pydpc/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/cwehmeyer/pydpc/actions/workflows/github-code-scanning/codeql)
[![Docs](https://github.com/cwehmeyer/pydpc/actions/workflows/docs.yml/badge.svg)](https://cwehmeyer.github.io/pydpc/)
[![PyPI version](https://badge.fury.io/py/pydpc.svg)](https://pypi.python.org/pypi/pydpc)
[![PyPI downloads](https://img.shields.io/pypi/dm/pydpc.svg)](https://pypi.python.org/pypi/pydpc)

*Clustering by fast search and find of density peaks* was designed by Alex Rodriguez and Alessandro Laio; see their [project page](http://people.sissa.it/~laio/Research/Res_clustering.php) for more information.

The pydpc package aims to make this algorithm available for Python users.

### Installation

Install pydpc via pip from the Python package index

```bash
pip install pydpc
```

or the latest version from github

```bash
pip install git+https://github.com/cwehmeyer/pydpc.git@master
```

### Quick start

```python
import numpy as np
from pydpc import Cluster

# a simple bimodal data set: two gaussian blobs centered at x=-4 and x=+4
npoints = 1000
points = np.random.randn(npoints, 2)
points[:, 0] += 4 * np.random.choice([-1, 1], size=npoints)

# computes distances, density, and delta, then shows the decision graph
clu = Cluster(points)

# pick outliers in the decision graph as cluster centers and assign points
clu.assign(min_density=25, min_delta=6)

clu.membership   # cluster index for each point
clu.core_idx     # indices of high-confidence ("core") points
clu.halo_idx     # indices of low-confidence ("halo") points
```

See [`docs/examples/Example01.ipynb`](docs/examples/Example01.ipynb) for a full walkthrough with plots.

See [`CHANGELOG.md`](CHANGELOG.md) for release history.
