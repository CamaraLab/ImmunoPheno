from .stvea_class import STvEA
from .stvea_data_processor import DataProcessor
from .stvea_cluster import Cluster
from .stvea_mapping import Mapping
from .stvea_annotation import Annotation
from copy import deepcopy
import pandas as pd
import numpy as np

class Controller:
    stvea = STvEA()
    data_processor = None
    cluster = None
    mapping = None
    annotation = None

    def __init__(self):
        self.stvea = STvEA()
        self.data_processor = DataProcessor(self.stvea)
        self.cluster = Cluster(self.stvea)
        self.mapping = Mapping(self.stvea)
        self.annotation = Annotation(self.stvea)

    def interface(self,
                 codex_protein,
                 cite_protein,
                 cite_cluster,
                 # map_codex_to_cite args
                 k_find_nn=80,
                 k_find_anchor=20,
                 k_filter_anchor=100,
                 k_score_anchor=80,
                 k_find_weights=100,
                 seed=42,
                 # parallelization
                 num_chunks=1,
                 num_cores=1,
                 # transfer_matrix
                 k_transfer_matrix=None,
                 c_transfer_matrix=0.1,
                 mask_threshold=0.5,
                 mask=True
                 ):
        """
        This function interface STvEA with database.
        The output of this function is stored in stvea.codex_cluster_names_transferred.

        @param codex_protein: a string to specify the address of CODEX protein dataset.
        @param cite_protein: a string to specify the address of CITE-seq protein dataset.
        @param k_find_nn: the number of nearest neighbors.
        @param k_find_anchor: The number of neighbors to find anchors.
            Fewer k_anchor should mean higher quality of anchors.
        @param k_filter_anchor: the number of nearest neighbors to find in the original data space.
        @param k_score_anchor: the number of nearest neighbors to use in shared nearest neighbor scoring.
        @param k_find_weights: the number of nearest anchors to use in correction.
        @param k_transfer_matrix: the number of nearest anchors to use in correction.
        @param c_transfer_matrix: a constant that controls the width of the Gaussian kernel.
        @param mask: a boolean value to specify whether to discard CODEX cells that don't have near CITE-seq cells.
        """
        # read and clean data
        self.stvea.codex_protein = codex_protein
        self.stvea.cite_protein = cite_protein
        self.stvea.cite_cluster = cite_cluster

        # map the CODEX cells to CITE-seq cells.
        self.mapping.map_codex_to_cite(k_find_nn=k_find_nn,
                                       k_find_anchor=k_find_anchor,
                                       k_filter_anchor=k_filter_anchor,
                                       k_score_anchor=k_score_anchor,
                                       k_find_weights=k_find_weights,
                                       num_chunks=num_chunks,
                                       num_cores=num_cores,
                                       seed=seed)

        # create transfer matrix to transfer values from CITE-seq to CODEX
        self.mapping.transfer_matrix(k=k_transfer_matrix,
                                     c=c_transfer_matrix,
                                     mask_threshold=mask_threshold)

        self.annotation.transfer_labels(user_input=False)

        # mask some CODEX cells that don't have near CITE-cells.
        if mask:
            self.data_processor.mask_codex(stvea=self.stvea)



    def pipeline2(self,
                 # read_codex args
                 codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                 codex_protein="../Data/raw_dataset/codex_protein.csv",
                 codex_size="../Data/raw_dataset/codex_size.csv",
                 codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                 codex_preprocess=True,
                 codex_border=564000,
                 # read_cite args
                 cite_latent="../Data/raw_dataset/cite_latent.csv",
                 cite_protein="../Data/raw_dataset/cite_protein.csv",
                 cite_mrna="../Data/raw_dataset/cite_mRNA.csv",
                 # take_subset args
                 amount_codex=-1,
                 amount_cite=-1,
                 # filter_codex args
                 size_lim=(1000, 25000),
                 blank_lower=(-1200, -1200, -1200, -1200),
                 blank_upper=(6000, 2500, 5000, 2500),
                 # clean_cite args
                 maxit=500,
                 factr=1e-9,
                 optim_init=([10, 60, 2, 0.5, 0.5],
                             [4.8, 50, 0.5, 2, 0.5],
                             [2, 18, 0.5, 2, 0.5],
                             [1, 3, 2, 2, 0.5],
                             [1, 3, 0.5, 2, 0.5]),
                 ignore_warnings=True,
                 clean_cite_method="l-bfgs-b",
                 # cluster_codex args
                 cluster_codex_k=30,
                 cluster_codex_option=1,
                 cluster_codex_threshold=(0.01, 0.001, 0.01, 0.01),
                 markers=("B220", "Ly6G", "NKp46", "TCR"),
                 # parameter_scan args
                 cluster_cite_option=1,
                 parameter_scan_min_cluster_size_range=tuple(range(5, 21, 4)),
                 parameter_scan_min_sample_range=tuple(range(10, 41, 3)),
                 parameter_scan_n_neighbors=50,
                 parameter_scan_min_dist=0.1,
                 parameter_scan_negative_sample_rate=50,
                 parameter_scan_metric="correlation",
                 silhoutte_metric="correlation",
                 # consensus_cluster args
                 consensus_cluster_silhouette_cutoff=0.114,
                 consensus_cluster_inconsistent_value=0.1,
                 consensus_cluster_min_cluster_size=10,
                 silhouette_cutoff_percentile=95,
                 # cite hdbscan
                 hdbscan_min_cluster_size_range=2,
                 hdbscan_min_sample_range=14,
                 hdbscan_n_neighbors=50,
                 hdbscan_min_dist=0.1,
                 hdbscan_negative_sample_rate=50,
                 hdbscan_cluster_metric="correlation",
                 hdbscan_random_state=0,
                 # map_codex_to_cite args
                 k_find_nn=80,
                 k_find_anchor=20,
                 k_filter_anchor=100,
                 k_score_anchor=80,
                 k_find_weights=100,
                 # transfer_matrix
                 k_transfer_matrix=None,
                 c_transfer_matrix=0.1,
                 mask_threshold=0.5,
                 mask=True
                 ):
        """
        This pipeline is for evaluation only.
        It will read CITE-seq annotations using original dataset.

        @param mask: a boolean value to specify whether to discard CODEX cells that don't have near CITE-seq cells.
        @param codex_blanks: a string to specify the address of CODEX blank dataset.
        @param codex_protein: a string to specify the address of CODEX protein dataset.
        @param codex_size: a string to specify the address of CODEX size dataset.
        @param codex_spatial: a string to specify the address of CODEX spatial dataset.
        @param codex_preprocess: a boolean value to specify whether to preprocess data as it is needed for raw data.
            Preprocess means to convert voxel to nm.
        @param codex_border: CODEX cells whose x and y are below this border will be kept in nm sense.
            564000 in nm sense is equivalent to 30000 in voxel sense.
            -1 means all CODEX cells will be kept.
        @param cite_latent: a string to specify the address of CITE-seq latent dataset.
        @param cite_protein: a string to specify the address of CITE-seq protein dataset.
        @param cite_mrna: a string to specify the address of CITE-seq mRNA file.
        @param amount_codex: the number of records will be kept for CODEX.
        @param amount_cite: the number of records will be kept for CITE_seq.
        @param size_lim: a size limit, default to (1000, 25000)
        @param blank_lower: a vector of length 4, default to (-1200, -1200, -1200, -1200)
        @param blank_upper: a vector of length 4, default to (6000, 2500, 5000, 2500)
        @param maxit: maximum number of iterations for optim function.
        @param factr: accuracy of optim function.
        @param optim_init: a ndarray of optional initialization parameters for the optim function,
            if NULL, starts at five default parameter sets and picks the better one.
            Sometimes, negative binomial doesn't fit well with certain starting parameters, so try 5.
        @param ignore_warnings: a boolean value to specify whether to ignore warnings or not.
        @param clean_cite_method: a string to specify the method that will be used to fit the mixture binomial distribution.
        @param cluster_codex_k: the number of nearest neighbors to generate graph.
            The graph will be used to perform Louvain community detection.
        @param cluster_codex_option: the way to detect the nearest neighbors.
            1: use Pearson distance to find the nearest neighbors on CODEX protein data.
            2: use Euclidean distance to find the nearest neighbors on 2D CODEX embedding data.
        @param parameter_scan_min_cluster_size_range: a vector of min_cluster_size arguments to scan over.
        @param parameter_scan_min_sample_range: a vector of min_sample arguments to scan over.
        @param parameter_scan_n_neighbors: the number of neighbors.
        @param parameter_scan_min_dist: the effective minimum distance between embedded points.
        @param parameter_scan_negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        @param parameter_scan_metric: Pearson correlation should be used here.
        @param consensus_cluster_silhouette_cutoff: HDBSCAN results below this cutoff will be discarded.
        @param consensus_cluster_inconsistent_value: input parameter to fcluster determining where clusters are cut in the hierarchical tree.
        @param consensus_cluster_min_cluster_size: cells in clusters smaller than this value are assigned a cluster ID of -1, indicating no cluster assignment.
        @param k_find_nn: the number of nearest neighbors.
        @param k_find_anchor: The number of neighbors to find anchors.
            Fewer k_anchor should mean higher quality of anchors.
        @param k_filter_anchor: the number of nearest neighbors to find in the original data space.
        @param k_score_anchor: the number of nearest neighbors to use in shared nearest neighbor scoring.
        @param k_find_weights: the number of nearest anchors to use in correction.
        @param k_transfer_matrix: the number of nearest anchors to use in correction.
        @param c_transfer_matrix: a constant that controls the width of the Gaussian kernel.
        """
        # read and clean data
        self.data_processor.read_codex(codex_blanks=codex_blanks,
                                       codex_protein=codex_protein,
                                       codex_size=codex_size,
                                       codex_spatial=codex_spatial,
                                       codex_preprocess=codex_preprocess,
                                       codex_border=codex_border
                                       )
        self.data_processor.read_cite(cite_latent=cite_latent,
                                      cite_protein=cite_protein,
                                      cite_mrna=cite_mrna)

        # take a certain number of cells
        self.data_processor.take_subset(amount_codex=amount_codex,
                                        amount_cite=amount_cite)
        self.data_processor.filter_codex(size_lim=size_lim,
                                         blank_lower=blank_lower,
                                         blank_upper=blank_upper)

        # clean CODEX cells and CITE-seq cells
        # self.data_processor.clean_codex()
        # self.data_processor.clean_cite(maxit=maxit,
        #                                factr=factr,
        #                                optim_init=optim_init,
        #                                ignore_warnings=ignore_warnings,
        #                                method=clean_cite_method)

        # map the CODEX cells to CITE-seq cells.
        self.mapping.map_codex_to_cite(k_find_nn=k_find_nn,
                                       k_find_anchor=k_find_anchor,
                                       k_filter_anchor=k_filter_anchor,
                                       k_score_anchor=k_score_anchor,
                                       k_find_weights=k_find_weights)

        non_na_row = ~ self.stvea.codex_protein_corrected.isna().any(axis=1)
        self.stvea.codex_protein_corrected = self.stvea.codex_protein_corrected.loc[non_na_row]
        self.stvea.codex_protein = self.stvea.codex_protein.loc[non_na_row]
        self.stvea.codex_size = self.stvea.codex_size.loc[non_na_row]
        self.stvea.codex_spatial = self.stvea.codex_spatial.loc[non_na_row]
        self.stvea.codex_blanks = self.stvea.codex_blanks.loc[non_na_row]

        # create transfer matrix to transfer values from CITE-seq to CODEX
        self.mapping.transfer_matrix(k=k_transfer_matrix,
                                     c=c_transfer_matrix,
                                     mask_threshold=mask_threshold)

        # self.stvea.codex_mask = pd.read_csv('../Data/immunopheno/codex_mask.csv', header=0, index_col=0)

        # remove some CODEX cells that don't have near CITE-cells.
        if mask:
            self.data_processor.discard_codex(stvea=self.stvea)

        # cluster CODEX cells
        self.cluster.cluster_codex(k=cluster_codex_k,
                                   option=cluster_codex_option,
                                   threshold=cluster_codex_threshold,
                                   markers=markers,
                                   plot_umap=True)

        self.stvea.cite_cluster = pd.read_csv("../Data/immunopheno/cite_cluster_labels.csv", index_col=0, header=0)



    def pipeline(self,
                 # read_codex args
                 codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                 codex_protein="../Data/raw_dataset/codex_protein.csv",
                 codex_size="../Data/raw_dataset/codex_size.csv",
                 codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                 codex_preprocess=True,
                 codex_border=564000,
                 # read_cite args
                 cite_latent="../Data/raw_dataset/cite_latent.csv",
                 cite_protein="../Data/raw_dataset/cite_protein.csv",
                 cite_mrna="../Data/raw_dataset/cite_mRNA.csv",
                 # take_subset args
                 amount_codex=-1,
                 amount_cite=-1,
                 # filter_codex args
                 size_lim=(1000, 25000),
                 blank_lower=(-1200, -1200, -1200, -1200),
                 blank_upper=(6000, 2500, 5000, 2500),
                 # clean_cite args
                 maxit=500,
                 factr=1e-9,
                 optim_init=([10, 60, 2, 0.5, 0.5],
                             [4.8, 50, 0.5, 2, 0.5],
                             [2, 18, 0.5, 2, 0.5],
                             [1, 3, 2, 2, 0.5],
                             [1, 3, 0.5, 2, 0.5]),
                 ignore_warnings=True,
                 clean_cite_method="l-bfgs-b",
                 # cluster_codex args
                 cluster_codex_k=30,
                 cluster_codex_option=1,
                 cluster_codex_threshold=(0.01, 0.001, 0.01, 0.01),
                 markers=("B220", "Ly6G", "NKp46", "TCR"),
                 # parameter_scan args
                 cluster_cite_option=1,
                 parameter_scan_min_cluster_size_range=tuple(range(5, 21, 4)),
                 parameter_scan_min_sample_range=tuple(range(10, 41, 3)),
                 parameter_scan_n_neighbors=50,
                 parameter_scan_min_dist=0.1,
                 parameter_scan_negative_sample_rate=50,
                 parameter_scan_metric="correlation",
                 silhoutte_metric="correlation",
                 # consensus_cluster args
                 consensus_cluster_silhouette_cutoff=0.114,
                 consensus_cluster_inconsistent_value=0.1,
                 consensus_cluster_min_cluster_size=10,
                 silhouette_cutoff_percentile=95,
                 # cite hdbscan
                 hdbscan_min_cluster_size_range=2,
                 hdbscan_min_sample_range=14,
                 hdbscan_n_neighbors=50,
                 hdbscan_min_dist=0.1,
                 hdbscan_negative_sample_rate=50,
                 hdbscan_cluster_metric="correlation",
                 hdbscan_random_state=0,
                 # map_codex_to_cite args
                 k_find_nn=80,
                 k_find_anchor=20,
                 k_filter_anchor=100,
                 k_score_anchor=80,
                 k_find_weights=100,
                 # transfer_matrix
                 k_transfer_matrix=None,
                 c_transfer_matrix=0.1,
                 mask_threshold=0.5,
                 mask=True
                 ):
        """
        This is the ultimate pipeline of STvEA to transfer labels from CITE-seq data to CODEX data.

        @param mask: a boolean value to specify whether to discard CODEX cells that don't have near CITE-seq cells.
        @param codex_blanks: a string to specify the address of CODEX blank dataset.
        @param codex_protein: a string to specify the address of CODEX protein dataset.
        @param codex_size: a string to specify the address of CODEX size dataset.
        @param codex_spatial: a string to specify the address of CODEX spatial dataset.
        @param codex_preprocess: a boolean value to specify whether to preprocess data as it is needed for raw data.
            Preprocess means to convert voxel to nm.
        @param codex_border: CODEX cells whose x and y are below this border will be kept in nm sense.
            564000 in nm sense is equivalent to 30000 in voxel sense.
            -1 means all CODEX cells will be kept.
        @param cite_latent: a string to specify the address of CITE-seq latent dataset.
        @param cite_protein: a string to specify the address of CITE-seq protein dataset.
        @param cite_mrna: a string to specify the address of CITE-seq mRNA file.
        @param amount_codex: the number of records will be kept for CODEX.
        @param amount_cite: the number of records will be kept for CITE_seq.
        @param size_lim: a size limit, default to (1000, 25000)
        @param blank_lower: a vector of length 4, default to (-1200, -1200, -1200, -1200)
        @param blank_upper: a vector of length 4, default to (6000, 2500, 5000, 2500)
        @param maxit: maximum number of iterations for optim function.
        @param factr: accuracy of optim function.
        @param optim_init: a ndarray of optional initialization parameters for the optim function,
            if NULL, starts at five default parameter sets and picks the better one.
            Sometimes, negative binomial doesn't fit well with certain starting parameters, so try 5.
        @param ignore_warnings: a boolean value to specify whether to ignore warnings or not.
        @param clean_cite_method: a string to specify the method that will be used to fit the mixture binomial distribution.
        @param cluster_codex_k: the number of nearest neighbors to generate graph.
            The graph will be used to perform Louvain community detection.
        @param cluster_codex_option: the way to detect the nearest neighbors.
            1: use Pearson distance to find the nearest neighbors on CODEX protein data.
            2: use Euclidean distance to find the nearest neighbors on 2D CODEX embedding data.
        @param parameter_scan_min_cluster_size_range: a vector of min_cluster_size arguments to scan over.
        @param parameter_scan_min_sample_range: a vector of min_sample arguments to scan over.
        @param parameter_scan_n_neighbors: the number of neighbors.
        @param parameter_scan_min_dist: the effective minimum distance between embedded points.
        @param parameter_scan_negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        @param parameter_scan_metric: Pearson correlation should be used here.
        @param consensus_cluster_silhouette_cutoff: HDBSCAN results below this cutoff will be discarded.
        @param consensus_cluster_inconsistent_value: input parameter to fcluster determining where clusters are cut in the hierarchical tree.
        @param consensus_cluster_min_cluster_size: cells in clusters smaller than this value are assigned a cluster ID of -1, indicating no cluster assignment.
        @param k_find_nn: the number of nearest neighbors.
        @param k_find_anchor: The number of neighbors to find anchors.
            Fewer k_anchor should mean higher quality of anchors.
        @param k_filter_anchor: the number of nearest neighbors to find in the original data space.
        @param k_score_anchor: the number of nearest neighbors to use in shared nearest neighbor scoring.
        @param k_find_weights: the number of nearest anchors to use in correction.
        @param k_transfer_matrix: the number of nearest anchors to use in correction.
        @param c_transfer_matrix: a constant that controls the width of the Gaussian kernel.
        """
        # read and clean data
        self.data_processor.read_codex(codex_blanks=codex_blanks,
                                       codex_protein=codex_protein,
                                       codex_size=codex_size,
                                       codex_spatial=codex_spatial,
                                       codex_preprocess=codex_preprocess,
                                       codex_border=codex_border
                                       )
        self.data_processor.read_cite(cite_latent=cite_latent,
                                      cite_protein=cite_protein,
                                      cite_mrna=cite_mrna)

        # take a certain number of cells
        self.data_processor.take_subset(amount_codex=amount_codex,
                                        amount_cite=amount_cite)

        self.data_processor.filter_codex(size_lim=size_lim,
                                         blank_lower=blank_lower,
                                         blank_upper=blank_upper)

        # clean CODEX cells and CITE-seq cells
        self.data_processor.clean_codex()

        self.data_processor.clean_cite(maxit=maxit,
                                       factr=factr,
                                       optim_init=optim_init,
                                       ignore_warnings=ignore_warnings,
                                       method=clean_cite_method)

        # map the CODEX cells to CITE-seq cells.
        self.mapping.map_codex_to_cite(k_find_nn=k_find_nn,
                                       k_find_anchor=k_find_anchor,
                                       k_filter_anchor=k_filter_anchor,
                                       k_score_anchor=k_score_anchor,
                                       k_find_weights=k_find_weights)

        # create transfer matrix to transfer values from CITE-seq to CODEX
        self.mapping.transfer_matrix(k=k_transfer_matrix,
                                     c=c_transfer_matrix,
                                     mask_threshold=mask_threshold)

        # remove some CODEX cells that don't have near CITE-cells.
        if mask:
            self.data_processor.discard_codex(stvea=self.stvea)

        # cluster CODEX cells
        self.cluster.cluster_codex(k=cluster_codex_k,
                                   option=cluster_codex_option,
                                   threshold=cluster_codex_threshold,
                                   markers=markers,
                                   plot_umap=True)

        # cluster CITE-seq cells
        self.cluster.cluster_cite(option=cluster_cite_option,
                                  # for parameter scan
                                  parameter_scan_min_cluster_size_range=parameter_scan_min_cluster_size_range,
                                  parameter_scan_min_sample_range=parameter_scan_min_sample_range,
                                  parameter_scan_n_neighbors=parameter_scan_n_neighbors,
                                  parameter_scan_min_dist=parameter_scan_min_dist,
                                  parameter_scan_negative_sample_rate=parameter_scan_negative_sample_rate,
                                  parameter_scan_cluster_metric=parameter_scan_metric,
                                  parameter_scan_silhoutte_metric=silhoutte_metric,
                                  parameter_scan_random_state=0,
                                  # for consensus cluster
                                  consensus_cluster_silhouette_cutoff=consensus_cluster_silhouette_cutoff,
                                  consensus_cluster_silhouette_cutoff_percentile=silhouette_cutoff_percentile,
                                  consensus_cluster_inconsistent_value=consensus_cluster_inconsistent_value,
                                  consensus_cluster_min_cluster_size=consensus_cluster_min_cluster_size,
                                  # hdbscan
                                  hdbscan_min_cluster_size_range=hdbscan_min_cluster_size_range,
                                  hdbscan_min_sample_range=hdbscan_min_sample_range,
                                  hdbscan_n_neighbors=hdbscan_n_neighbors,
                                  hdbscan_min_dist=hdbscan_min_dist,
                                  hdbscan_negative_sample_rate=hdbscan_negative_sample_rate,
                                  hdbscan_cluster_metric=hdbscan_cluster_metric,
                                  hdbscan_random_state=hdbscan_random_state,
                                  # plot
                                  plot_umap=True)
