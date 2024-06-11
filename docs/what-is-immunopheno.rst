What is ImmunoPheno?
====================

ImmunoPheno is a Python library that aims to improve immunophenotyping across
multiple types of experiments. This is valuable in cancer research,
where understanding the tumor microenvironment can help with analyzing cancer progression,
treatment, and metastasis. However, current multiplexed antibody-based cytometry techniques
used for these experiments face significant challenges. One of these challeges is the manual and subjective
process of annotating cell populations, which affects accuracy and reproducibility. Another challenge is
the difficulty of distinguishing cell clusters due to overlapping antigenic profiles. Lastly, creating
antibody panels that include markers for all cell types and states is impractical, especially if 
rare populations are present in the sample.

To address these issues, ImmunoPheno uses existing single cell atlases to
build a comprehensive reference antigenic dataset that contains information from
a wide range of experiments. These experiments contain different types of antibodies, tissue samples,
and cell populations. As a result, this reference dataset can be used to automatically annotate and gate
cell populations in cytometry data. Moreover, it can also assist in the design of optimal antibody panels
tailored to specific tissues or sets of cell populations.

