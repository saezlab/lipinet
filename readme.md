# LipiNet (Lipids PKN)*
*_(The LipiNet name is a placeholder for now, might be changed down the line)_

## Context

The lipidomics field faces unique challenges in standardizing its nomenclature and measurement precision, unlike genomics, transcriptomics, and proteomics, which have relatively consistent units of measurement (genes, transcripts, proteins). In lipidomics, measurement limitations frequently prevent analysts from identifying lipids at precise structural or isomeric subspecies levels. Consequently, lipid identification often relies on generalized representations, such as abstract class or species names aligned with established ontologies. This, along with variations in database standards, creates a particularly fragmented and complex landscape for prior knowledge in lipidomics.

LipiNet is designed to address these challenges by integrating information across disparate lipidomics databases, each with different identifiers and varying levels of lipid resolution. By unifying these resources and accounting for the inherent ambiguity in lipid identification, LipiNet enables more cohesive and comprehensive network analyses across lipidomics databases.

## Core features 

- Multi-layered network construction and analysis 
- Cross-database lipid identifier integration 
- Tools for node property propagation, community detection, network-based link inference, and modelling with prior knowledge 

## Getting started

You can currently install using pip via GitHub, or alternatively through local installation.

GitHub
1. `pip install git+https://github.com/saezlab/lipinet.git`

Local installation 
1. download locally and navigate to the directory where `setup.py` is located
2. install as an editable package: `pip install -e .`

## Roadmap 

Phase 1: Core network development 
1. Establish network architecture 
2. Define node properties 

Phase 2: Basic analysis functionality 
1. Filtering capabilities 
2. Path condensation

Phase 3: Advanced network and structural analysis 
1. Network transformation
2. Missing link inference 
3. Community detection

Phase 4: Measured data integration and statistical modelling 
1. Node property categorisation and conditional inference 
2. Statistical modelling 

## Usage examples 

