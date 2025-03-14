# setup.py
from setuptools import setup, find_packages

setup(
    name='lipinet',
    version='1.0.0',
    packages=find_packages(),
)

# if editing the package and testing, you should run this in the root directory (LipiNet):
#   pip install -e .
# to uninstall, use:    pip uninstall
# or to reinstall, without edit mode, use: pip install .