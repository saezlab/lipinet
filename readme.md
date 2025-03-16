# LipiNet <img src="./.assets/.lipinet_logo_v1_0051.png" alt="OnionNet Logo" width="140" align="right" />

## Context

The lipidomics field faces unique challenges in standardizing its nomenclature and measurement precision, unlike genomics, transcriptomics, and proteomics, which have relatively consistent units of measurement (genes, transcripts, proteins). In lipidomics, measurement limitations frequently prevent analysts from identifying lipids at precise structural or isomeric subspecies levels. Consequently, lipid identification often relies on generalized representations, such as abstract class or species names aligned with established ontologies. This, along with variations in database standards, creates a particularly fragmented and complex landscape for prior knowledge in lipidomics.

LipiNet is designed to address these challenges by integrating information across disparate lipidomics databases, each with different identifiers and varying levels of lipid resolution. By unifying these resources and accounting for the inherent ambiguity in lipid identification, LipiNet enables more cohesive and comprehensive network analyses across lipidomics databases.

## Core features 

- Multi-layered network construction and analysis 
- Cross-database lipid identifier integration 
- Tools for filtering, analysing and visualising by layers

## Installation

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

## Quick Start

The general framework for LipiNet is:
1. Dataset parsing
2. Dataset exploration
3. Integration (in progress)

You can currently find tutorials for the first two steps with SwissLipids in the notebooks folder, describing the network creation process and initial findings from exploration.

## License

Most of the datasets in `LipiNet` are openly available for use provided you cite them accordingly. We encourage users to check the terms of the resource themselves.

## Contributing 

Both `LipiNet` and `OnionNet` are in active development and subject to change. Some functions may be modified or deprecated in future releases. If you find `LipiNet` helpful or have ideas for improvement, we'd love to hear more!