# Installation

`LipiNet` relies on functions in `OnionNet`, which in turn requires `graph-tool` to be installed. Because `graph-tool` is built around C++ for efficiency, unfortunately there is no straightforward pip installation. Nonetheless, there are a number of ways to install `graph-tool` besides pip, see [here](https://graph-tool.skewed.de/installation.html) for more details. The easiest way for most users is probably to create a new env via `conda`:

```
conda create --name gt -c conda-forge graph-tool ipython jupyter
conda activate gt
```
Then you can install `OnionNet` within the conda env with:
```
pip install git+https://github.com/saezlab/onionnet.git
```
Finally you can install `LipiNet` in a similar fashion:
```
pip install git+https://github.com/saezlab/lipinet.git
```
Now you should be ready to go!

In the near future we intend to include both `LipiNet` and `OnionNet` on PyPI.