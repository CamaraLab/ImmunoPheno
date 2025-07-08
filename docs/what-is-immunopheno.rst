What is ImmunoPheno?
====================

Immunophenotyping is widely used to characterize the cellular composition of tissues by 
measuring cell surface protein markers. Current techniques, such as highly 
multiplexed cytometry and immunohistochemistry, enable the simultaneous profiling of dozens 
of markers using antibody panels. However, the full potential of these approaches is often limited
by significant challenges in both experimental design and data analysis.

The complexity and cellular heterogeneity of human tissues, and particularly cancer tissues, make it 
difficult to design antibody panels that comprehensively capture all relevant cell types and states,
especially rare cell populations. As a result, different cell populations may exhibit only subtle
differences in their protein expression depending on the antibody panel used. Moreover, 
standard analysis methods often involve manual gating to identify cell populations, which is 
time-consuming, subjective, and poorly reproducible.

To address these limitations, we have developed ImmunoPheno, a Python library and online 
resource for the automated design of antibody panels and the identification and annotation of cell 
populations in immunophenotyping data. Built on our prior work STvEA (Spatial 
Transcriptomics via Epitope Anchoring), ImmunoPheno leverages harmonized reference single-
cell proteotranscriptomic data (e.g., CITE-seq, Abseq, REAP-seq) to detect subtle cell populations, spatial patterns of transcription, 
and potential paracrine interactions in multiplexed antibody-based cytometry data. 

To support ImmunoPheno, we created the Human Immune Cell Antibody Reference 
(HICAR), a comprehensive reference atlas constructed by harmonizing both public and newly 
generated single-cell proteo-transcriptomic data. HICAR currently includes harmonized protein expression
profiles for 390 monoclonal antibodies across 93 immune cell types.

ImmunoPheno uses this resource to overcome key bottlenecks in cytometry. It streamlines 
experimental design by recommending minimal antibody panels that distinguish target cell 
populations and suggesting optimized gating strategies. For data analysis, it provides rapid, 
automated, and reproducible cell identity annotation for new cytometry experiments.

By addressing challenges in both experimental design and data analysis, ImmunoPheno 
enables more rigorous and reproducible immunophenotyping, improve the phenotypic resolution, accuracy, and 
reproducibility of cytometry experiments.

The installation instructions and source code of ImmunoPheno can be found in the `Github site <https://github.com/CamaraLab/ImmunoPheno>`_.

A web app demonstrating some of ImmunoPheno's functinalities is available at `http://immunopheno.org/ <http://immunopheno.org>`_.
