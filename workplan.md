# Workplan for LipiNet (Lipids PKN)*
*_(The LipiNet name is a placeholder for now, might be changed down the line)_

## Context

The state of the lipidomics field is currently quite unique in terms of its ambiguity. Whereas in genomics, transcriptomics, and proteomics, there is a fairly unambiguous nomenclature relating to the individual units of measurement (i.e. genes, transcripts, and protein). Although there is still some level of variation within all of these terms, this is relatively small compared to lipidomics, where measurement limitations often hinder the analysts ability to establish what the precise lipid really is at the isomeric subspecies or structural level.  Instead, it is very common within the field to resort to a higher level representation of the lipid, for example by using a more abstract class or species name of a lipid according to established ontologies. This, in addition to the problems with standardising various databases and resources, leads to a prior knowledge landscape that is especially confusing and messy for lipidomics. 

The prior knowledge for lipidomics is splintered across different databases, with limited shared identifiers, and representing lipids at varying levels of resolution. This makes it challenging to perform network analyses that integrate the information across different databases. LipiNet is being developed to overcome this, to integrate across the different databases, whilst taking into account the ambiguity of lipid resolution.

## Network fundamentals

#### Data structure of the multilayered network

There are many competing architectures we could use to build the multilayered network(/s), with pros and cons for each:
1. Deciding upon a single type of lipid identifier to use (probably based on an ontology e.g. SwissLipids, LipidMaps, etc.), then creating each layer at that lipid type level and mapping onto it from the other sources (e.g. mapping Rhea reactions to ChEBI)
    - Pros: 
      - standardised networks across all levels
      - reduces amount of work to include multiple sources and complex representations
      - simplifies downstream functions and analysis 
    - Cons: 
      - if mappings to chosen lipid identifier are absent (e.g. no ChEBI/Rhea mapping to SwissLipids), then could lose information or bias towards canonical prior knowledge 
        - on the other hand, could be partially overcome perhaps with extra layers mapping lipids to canonical IDs
2. Represent each layer 'as-is' according to the data source (as much as can be done)(e.g. SwissLipids: entire ontology, Rhea: all reactions, Reactome: all pathways). Then create functionality to map between these as best as possible.
    - Pros:
      - keeps more of the original data sources
      - more future friendly (perhaps), and easier for analyst or downstream development to 'pick and choose' layers 
    - Cons:
      - much higher complexity for each network, increases development time
3. Mixed approach 

#### Node properties 

Probably irrespective of the multilayered data structure we select, these are the data structures and properties we want the network to have:
- layers
- authority / original source of information 
- node property / type (generalised, e.g. ChEBI, Rhea ID, etc.)
- ID (individual node ID)

##### Layers
The layers should include:
- SwissLipids ontology
- LipidMaps ontology 
- Rhea reactions 
- Reactome
- ChEBI ontology 
- ClassyFire ontology 
- Structures (future development)


## Analysis functionality 

#### For prior knowledge

We need methods to:
1. efficiently filter (or colour) by node properties / layer
2. efficiently filter (or colour) by edge properties 
3. way to collapse / condense node paths by parents or children  

Some further nice functionalities, might be:
- convert certain layers into hypergraphs
- map to structural data, and perform structural clustering within the network or certain levels of the hierarchy, to create connections between canonical lipids with prior knowledge associations, and more obscure or unknown lipids (ManyZyme like function)
- infer missing links through network methods (ManyZyme like function)
- perform community detection on the overall network, or certain layers, to explore patterns or latent structure for example using nested SBM

#### For measured data

For each node, we should categorise the node properties as either:
1. inherited from parent
2. directly associated with node
3. inferred from children (with XYZ conditional probability)

Some further nice functionalities, might be:
- incorporate prior knowledge (about species, measurement instrument, frequency of certain lipids, etc.) into a statistical model (e.g. Bayesian hierarchical models)