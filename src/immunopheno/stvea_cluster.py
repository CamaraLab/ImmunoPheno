import math
import random
import time
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
import umap
import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from igraph import Graph
from numba import NumbaDeprecationWarning
import seaborn as sns
import math
from sklearn.decomposition import KernelPCA

class Cluster:
    stvea = None

    def __init__(self, stvea):
        self.stvea = stvea

    @staticmethod
    def add_to_edge_list(edge_list, row, row_name):
        """
        This is a helper function invoked by cluster_codex.

        @param edge_list: an edge list.
        @param row: a pandas' dataframe row.
        @param row_name: the row's name.
        """
        row.apply(lambda col: edge_list.append([row_name, col]))

    def cluster_codex(self,
                      k=30,
                      option=1,
                      random_state=0,
                      plot=False,
                      threshold=(0.01, 0.001, (0.01, 0.1), 0.01),
                      markers = ("B220", "Ly6G", ("NKp46", "CD45"), "TCR"),
                      plot_umap=False):
        """
        This function will cluster codex cells.

        @param random_state: an integer to specify the random state.
        @param k: the number of nearest neighbors to generate graph.
        The graph will be used to perform Louvain community detection.
        @param option: the way to perform clustering.
        1: use Pearson distance to find the nearest neighbors on CODEX protein data.
        2: use Euclidean distance to find the nearest neighbors on 2D CODEX embedding data.
        """
        start = time.time()
        random.seed(0)

        # find knn
        if option == 1 or option == 3:
            # use Pearson distance to find nearest neighbors on CODEX protein data.

            umap_results = umap.nearest_neighbors(X=self.stvea.codex_protein_corrected,
                                       metric="correlation",
                                       n_neighbors=k,
                                       metric_kwds={},
                                       random_state=random_state,
                                       angular=False)

            self.stvea.codex_knn = pd.DataFrame(umap_results[0])
            self.stvea.codex_knn = self.stvea.codex_knn.iloc[:, 1:]

            # convert to pandas dataframe
            codex_knn = pd.DataFrame(self.stvea.codex_knn)

            # create an edge list
            edge_list = []
            codex_knn.apply(lambda row: Cluster.add_to_edge_list(edge_list, row, row.name), axis=1)

            # perform louvain community detection ()
            g = Graph(edges=edge_list)

            # add one to make clusters 1-indexed
            self.stvea.codex_cluster = pd.DataFrame(g.community_multilevel().membership,
                                                    index=self.stvea.codex_protein_corrected.index) + 1

        if option == 2:
            # use parameter_scan and consensus cluster
            # the same approach to cluster CITE-seq cells
            self.parameter_scan(min_cluster_size_range=tuple(range(5, 21, 4)),
                                min_sample_range=tuple(range(10, 41, 3)),
                                n_neighbors=50,
                                min_dist=0.1,
                                negative_sample_rate=50,
                                metric="correlation",
                                random_state=0,
                                option=2)
            self.consensus_cluster(silhouette_cutoff=0,
                                   inconsistent_value=0.1,
                                   min_cluster_size=1,
                                   option=2)

            return

        if option == 3:
            # use HDBSCAN to filter out noise within each cluster
            # not very good result
            temp = deepcopy(self.stvea.codex_cluster)
            for each_cluster in self.stvea.codex_cluster[0].unique():
                subset_index = self.stvea.codex_cluster[0] == each_cluster
                protein_subset = self.stvea.codex_protein_corrected.loc[subset_index, :]
                total_sum = subset_index.sum()
                reducer = umap.UMAP(n_components=4,
                                    n_neighbors=20,
                                    metric="correlation",
                                    random_state=random_state,
                                    init="random")
                umap_latent = reducer.fit_transform(protein_subset)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=max(round((total_sum)/10), 2), min_samples=5, metric="correlation")
                labels = pd.Series(clusterer.fit_predict(umap_latent))
                labels.index = temp[subset_index].index
                labels_replaced = labels.apply(lambda x: -1 if x == -1 else each_cluster)
                labels_replaced.index = temp[subset_index].index
                temp.loc[subset_index, 0] = labels_replaced
                if plot == True:
                   # reducer = umap.UMAP(n_components=2,
                   #                     n_neighbors=20,
                   #                     random_state=random_state,
                   #                     init="random")
                   # for_plot = reducer.fit_transform(protein_subset)
                    pca = KernelPCA(n_components=2, kernel="linear")
                    for_plot = pca.fit_transform(protein_subset)
                    warnings.simplefilter("ignore")
                    transferred_name = self.stvea.codex_cluster_names_transferred.loc[subset_index, 0]
                    transferred_name = transferred_name.rename("Labels")
                    plt.figure(figsize=(6, 6))
                    x = pd.Series(for_plot[:, 0], name="X", index=transferred_name.index)
                    y = pd.Series(for_plot[:, 1], name="Y", index=transferred_name.index)
                    df1 = pd.concat([x, y, transferred_name], axis=1)
                    sns.scatterplot(data=df1, x="X", y="Y", hue="Labels", palette="deep", s=60)
                    plt.title(f'UMAP Projection for Cluster {each_cluster} (Transferred)')
                    plt.show()

                    original_name = temp.loc[subset_index, 0].rename("Labels")
                    df2 = pd.concat([x, y, original_name], axis=1)
                    plt.figure(figsize=(6, 6))
                    sns.scatterplot(data=df2, x="X", y="Y", hue="Labels", palette="deep", s=60)
                    plt.title(f'UMAP Projection for Cluster {each_cluster} (Reclusterred)')
                    plt.show()

            self.stvea.codex_cluster = temp

        if option == 4:
            self.stvea.codex_cluster = pd.DataFrame(-1, index=self.stvea.codex_protein.index, columns=range(1))
            # only focus on B cells, T cells, Neutrophils, and NK cells
            n_cells_total = self.stvea.codex_protein_corrected.shape[0]
            for index, marker in enumerate(markers):
                if isinstance(marker, tuple):
                    codex_cluster_copy = pd.Series(True, index=self.stvea.codex_protein.index)
                    for sub_threshold, sub_marker in zip(threshold[index], marker):
                        n_cells = math.floor(sub_threshold * n_cells_total)
                        temp = pd.Series(False, index=self.stvea.codex_protein.index)
                        temp.loc[self.stvea.codex_protein_corrected.nlargest(n_cells, sub_marker).index] = True
                        codex_cluster_copy = codex_cluster_copy & temp
                    self.stvea.codex_cluster.loc[codex_cluster_copy, :] = index + 1
                    continue
                n_cells = math.floor(threshold[index] * n_cells_total)
                marker_cells = self.stvea.codex_protein_corrected.nlargest(n_cells, marker).index
                self.stvea.codex_cluster.loc[marker_cells, :] = index + 1
            return

        if option == 5:
            data = self.stvea.codex_protein_corrected
            reducer = umap.UMAP(n_components=data.shape[1],
                                n_neighbors=15,
                                min_dist=0.1,
                                negative_sample_rate=50,
                                metric="correlation",
                                random_state=random_state)

            umap_latent = reducer.fit_transform(data)

            cluster = hdbscan.HDBSCAN(min_cluster_size=3,
                                      min_samples=20,
                                      metric="euclidean",
                                      memory='./HDBSCAN_cache')

            hdbscan_labels = cluster.fit_predict(umap_latent)

            self.stvea.codex_cluster = pd.DataFrame(hdbscan_labels)

        end = time.time()
        print(f"CODEX clusters found. Time: {round(end - start, 3)} sec")
        if plot_umap:
            self.plot_codex(n_neighbors=30)
        return


    def codex_umap(self,
                   metric="correlation",
                   n_neighbors=30,
                   min_dist=0.1,
                   negative_sample_rate=50,
                   ignore_warnings=True,
                   random_state=0):
        """
        This method performs umap on codex protein data and creates a 2d embedding.

        @param ignore_warnings: a boolean value to specify whether to ignore warnings or not.
        @param metric: the metric to use here, default to correlation.
        @param n_neighbors: the number of neighbors, default to 30.
        @param min_dist: the effective minimum distance between embedded points.
        @param negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        @param random_state: an integer to specify the random state.
        """
        if ignore_warnings:
            warnings.filterwarnings("ignore")
        if self.stvea.codex_protein.empty:
            raise ValueError("stvea object does not contain codex protein data")
        random.seed(0)
        warnings.filterwarnings("ignore")
        # create a 2D embedding
        res = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                        negative_sample_rate=negative_sample_rate, n_components=2,
                        random_state=random_state).fit_transform(self.stvea.codex_protein)
        # convert to Pandas dataframe
        self.stvea.codex_emb = pd.DataFrame(res, index=self.stvea.codex_protein.index)

        return

    def cite_umap(self,
                  metric="correlation",
                  n_neighbors=50,
                  min_dist=0.1,
                  negative_sample_rate=50,
                  random_state=0,
                  ignore_warnings=True):
        """
        This function will perform umap on cite_latent or cite_mrna data,
        and construct a 2D embedding.
        cite_latent and cite_mrna will be used here instead of protein data
        because cite_latent and cite_mrna are more informative

        @param ignore_warnings: a boolean value to specify whether to ignore warnings.
        @param metric: the metric to calculate distance.
        @param n_neighbors: the number of neighbors.
        @param min_dist: the effective minimum distance between embedded points.
        @param negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        """
        if ignore_warnings:
            warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)

        if self.stvea.cite_latent.empty and self.stvea.cite_mRNA.empty:
            raise ValueError("stvea object does not contain CITE-seq mRNA or latent data")

        if not self.stvea.cite_latent.empty:
            # recommended
            res = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, n_components=2, random_state=random_state).fit_transform(
                self.stvea.cite_latent)
            self.stvea.cite_emb = pd.DataFrame(res)

        elif not self.stvea.cite_mRNA.empty:
            # implemented, but not recommended
            res = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, n_components=2, random_state=random_state).fit_transform(
                self.stvea.cite_mRNA)
            self.stvea.cite_emb = pd.DataFrame(res)

        return

    def plot_codex(self,
                   n_neighbors=30):
        self.codex_umap(n_neighbors=n_neighbors)
        self.stvea.codex_emb.index = self.stvea.codex_cluster.index
        df = pd.DataFrame({'x': self.stvea.codex_emb.iloc[:, 0],
                           'y': self.stvea.codex_emb.iloc[:, 1],
                           'Clusters': self.stvea.codex_cluster.iloc[:, 0]})
        plt.figure(figsize=(12, 12));
        sns.scatterplot(data=df, x="x", y="y", hue="Clusters", palette="tab20", s=10)
        plt.title("CODEX cells umap plot")
        plt.show()

    def plot_cite(self):
        self.cite_umap();
        self.stvea.cite_emb.index = self.stvea.cite_cluster.index
        df = pd.DataFrame({'x': self.stvea.cite_emb.iloc[:, 0],
                           'y': self.stvea.cite_emb.iloc[:, 1],
                           'Clusters': self.stvea.cite_cluster.iloc[:, 0]})
        plt.figure(figsize=(12, 12));
        sns.scatterplot(data=df, x="x", y="y", hue="Clusters", palette="tab20", s=10)
        plt.title("CITE-seq cells umap plot")
        plt.show()


    @staticmethod
    def run_hdbscan(umap_latent,
                    min_cluster_size_list,
                    min_sample_list,
                    metric,
                    cache_dir="./HDBSCAN_cache"):
        """
        This function is borrowed from the original STvEA R library.
        https://github.com/CamaraLab/STvEA/blob/master/inst/python/consensus_clustering.py

        @param metric: metric to run HDBSCAN.
        @param umap_latent: cite_latent data after being transformed by umap.
        @param min_cluster_size_list: a vector of min_cluster_size arguments to scan over.
        @param min_sample_list: a vector of min_sample arguments to scan over.
        @param cache_dir: a folder to store intermediary results of HDBSCAN.
        @return: a list of assigned labels.
        """
        umap_latent = np.array(umap_latent)
        results = []
        for min_samples in min_sample_list:
            for min_cluster_size in min_cluster_size_list:
                if cache_dir is None:
                    cluster = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                                              metric=metric)
                else:
                    cluster = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                                              metric=metric, memory=cache_dir)
                hdbscan_labels = cluster.fit_predict(umap_latent)
                results.append(hdbscan_labels.tolist())
        return results

    def parameter_scan(self,
                       min_cluster_size_range=tuple(range(5, 21, 4)),
                       min_sample_range=tuple(range(10, 41, 3)),
                       n_neighbors=50,
                       min_dist=0.1,
                       negative_sample_rate=50,
                       cluster_metric="correlation",
                       silhoutte_metric="correlation",
                       random_state=0,
                       option=1):
        """
        This function will run HDBSCAN multiple times given the vector of min_cluster_size_range and min_sample_range.
        The result will be a list of dictionaries that will record each HDBSCAN's scores and generated labels.
        @param random_state: an integer to specify the random state.
        @param min_cluster_size_range: a vector of min_cluster_size arguments to scan over.
        @param min_sample_range: a vector of min_sample arguments to scan over.
        @param n_neighbors: the number of neighbors.
        @param min_dist: the effective minimum distance between embedded points.
        @param negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        @param metric: Pearson correlation should be used here.
        @return: a list of dictionaries that will record each HDBSCAN's scores and generated labels.
        """
        start = time.time()

        if option == 1:
            data = self.stvea.cite_latent
            title = "CITE-seq "
        else:
            data = self.stvea.codex_protein
            title = "CODEX "

        # running UMAP on the data
        reducer = umap.UMAP(n_components=data.shape[1],
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate,
                            metric=cluster_metric,
                            random_state=random_state)
        umap_latent = reducer.fit_transform(data)

        # running HDBSCAN on the UMAP space
        hdbscan_labels = Cluster.run_hdbscan(umap_latent, min_cluster_size_range, min_sample_range, metric=cluster_metric)

        # calculate scores
        all_scores = []
        hdbscan_results = []
        for label in hdbscan_labels:
            # calculate silhouette scores
            # score = silhouette_score(cite_latent, label, metric="correlation")
            scores = silhouette_samples(data, label, metric=silhoutte_metric)
            score = np.mean(scores)
            all_scores.append(score)
            hdbscan_results.append({
                "cluster_labels": label,
                "silhouette_score": score
            })

        print("all_scores", all_scores)
        # plotting histogram of all silhouette scores
        plt.hist(all_scores, bins=100)
        plt.title(title + "Histogram of silhouette scores")
        plt.xlabel("Silhouette score")
        plt.ylabel("Number of calls to HDBSCAN")
        plt.show()

        self.stvea.hdbscan_scans = hdbscan_results
        end = time.time()
        print(f"Parameter scan done. Time: {round(end - start, 3)} sec")
        return

    @staticmethod
    def consensus_cluster_internal(distance_matrix,
                                   inconsistent_value=0.3,
                                   min_cluster_size=10):
        """
        This function was borrowed from the original STvEA R program.
        This function will perform clustering given the distance matrix.
        https://github.com/CamaraLab/STvEA/blob/master/inst/python/consensus_clustering.py

        @param distance_matrix: a consensus distance matrix based on HDBSCAN results.
        @param inconsistent_value: input parameter to fcluster determining where clusters are cut in the hierarchical tree.
        @param min_cluster_size: cells in clusters smaller than this value are assigned a cluster ID of -1, indicating no cluster assignment.
        @return: a list of labels
        """

        new_distance = squareform(distance_matrix)
        hierarchical_tree = linkage(new_distance, "average")
        hier_consensus_labels = fcluster(hierarchical_tree, t=inconsistent_value)
        hier_unique_labels = set(hier_consensus_labels)
        for label in hier_unique_labels:
            indices = [i for i, x in enumerate(hier_consensus_labels) if x == label]
            if len(indices) < min_cluster_size:
                for index in indices:
                    hier_consensus_labels[index] = -1
        return pd.DataFrame(hier_consensus_labels.tolist())

    def consensus_cluster(self,
                          silhouette_cutoff,
                          silhouette_cutoff_percentile,
                          inconsistent_value,
                          min_cluster_size,
                          option=1):
        """
        This function will first generate a distance matrix based on HDBSCAN results,
        and then invoke clustering function to perform clustering based on the distance matrix.
        @param silhouette_cutoff: HDBSCAN results below this cutoff will be discarded.
        @param inconsistent_value: input parameter to fcluster determining where clusters are cut in the hierarchical tree.
        @param min_cluster_size: cells in clusters smaller than this value are assigned a cluster ID of -1, indicating no cluster assignment.
        """
        start = time.time()

        # initialize some variables
        hdbscan_results = self.stvea.hdbscan_scans
        num_cells = len(hdbscan_results[0]['cluster_labels'])
        # initialize a consensus matrix
        consensus_matrix = np.zeros((num_cells, num_cells))
        total_runs = 0

        if silhouette_cutoff == None:
            scores = [x['silhouette_score'] for x in hdbscan_results]
            silhouette_cutoff = np.percentile(scores, silhouette_cutoff_percentile)

        # scan each result in the hdbscan_results
        for result in hdbscan_results:

            # if the silhouette score is larger than the silhouette_cutoff
            if result['silhouette_score'] >= silhouette_cutoff:

                # calculate a similar matrix sim_matrix
                sim_matrix = np.zeros((num_cells, num_cells))

                # loop through each cell
                for cell1 in range(num_cells):

                    # if a cell belongs to the same cluster
                    # 1 in the entry
                    if result['cluster_labels'][cell1] != -1:
                        condition = 1 * (pd.Series(result['cluster_labels']) == result['cluster_labels'][cell1])
                        sim_matrix[:, cell1] = condition

                # set diagonal entries in the matrix to be 1
                np.fill_diagonal(sim_matrix, 1)

                # substract the results from the consensus_matrix
                consensus_matrix -= sim_matrix

                # update the total number of runs
                total_runs += 1

        # flip the result to become a distance matrix
        consensus_matrix += total_runs

        if total_runs > 0:
            # normalize
            consensus_matrix /= total_runs
        else:
            print("Warning: No clustering runs passed the silhouette score cutoff.")
            exit(1)

        # perform clustering
        consensus_clusters = Cluster.consensus_cluster_internal(consensus_matrix,
                                                                inconsistent_value,
                                                                min_cluster_size)
        # relabel the result as R implementation did
        # ravel() will flat the array into 1-d:
        # [[1], [2], [3]] will be flattened into [1, 2, 3]
        original_array = np.array(consensus_clusters).ravel()

        # find unique elements and sort them
        unique_elements = np.sort(np.unique(original_array))

        # create a pandas series with index as unique_elements and values as sequence from -1
        map_series = pd.Series(np.concatenate([np.arange(-1, 0), np.arange(1, len(unique_elements))]),
                               index=unique_elements)

        # map original_array to new values using map_series
        # store the result in the STvEA object
        if option == 1:
            self.stvea.cite_cluster = pd.DataFrame(map_series[original_array].values)
        else:
            self.stvea.codex_cluster = pd.DataFrame(map_series[original_array].values)

        end = time.time()
        print(f"Consensus cluster done. Time: {round(end - start, 3)} sec")

        return


    def cluster_cite(self,
                     option=1,
                     # for parameter scan
                     parameter_scan_min_cluster_size_range=tuple(range(5, 21, 4)),
                     parameter_scan_min_sample_range=tuple(range(10, 41, 3)),
                     parameter_scan_n_neighbors=50,
                     parameter_scan_min_dist=0.1,
                     parameter_scan_negative_sample_rate=50,
                     parameter_scan_cluster_metric="correlation",
                     parameter_scan_silhoutte_metric="correlation",
                     parameter_scan_random_state=0,
                     # for consensus cluster
                     consensus_cluster_silhouette_cutoff=None,
                     consensus_cluster_silhouette_cutoff_percentile=95,
                     consensus_cluster_inconsistent_value=0.1,
                     consensus_cluster_min_cluster_size=8,
                     consensus_cluster_option=1,
                     # hdbscan
                     hdbscan_min_cluster_size_range=2,
                     hdbscan_min_sample_range=14,
                     hdbscan_n_neighbors=50,
                     hdbscan_min_dist=0.1,
                     hdbscan_negative_sample_rate=50,
                     hdbscan_cluster_metric="correlation",
                     hdbscan_random_state=0,
                     # plot
                     plot_umap=False
                     ):

        if option == 1:
            # traditional way
            # parameter_scan + consensus_cluster
            self.parameter_scan(min_cluster_size_range=parameter_scan_min_cluster_size_range,
                                min_sample_range=parameter_scan_min_sample_range,
                                n_neighbors=parameter_scan_n_neighbors,
                                min_dist=parameter_scan_min_dist,
                                negative_sample_rate=parameter_scan_negative_sample_rate,
                                cluster_metric=parameter_scan_cluster_metric,
                                silhoutte_metric=parameter_scan_silhoutte_metric,
                                random_state=parameter_scan_random_state,
                                )
            self.consensus_cluster(silhouette_cutoff=consensus_cluster_silhouette_cutoff,
                              silhouette_cutoff_percentile=consensus_cluster_silhouette_cutoff_percentile,
                              inconsistent_value=consensus_cluster_inconsistent_value,
                              min_cluster_size=consensus_cluster_min_cluster_size,
                              option=consensus_cluster_option)
        elif option == 2:
            # parameter scan and pick the best one
            self.parameter_scan(min_cluster_size_range=parameter_scan_min_cluster_size_range,
                                min_sample_range=parameter_scan_min_sample_range,
                                n_neighbors=parameter_scan_n_neighbors,
                                min_dist=parameter_scan_min_dist,
                                negative_sample_rate=parameter_scan_negative_sample_rate,
                                cluster_metric=parameter_scan_cluster_metric,
                                silhoutte_metric=parameter_scan_silhoutte_metric,
                                random_state=parameter_scan_random_state,
                                )

            maximum_index = -1
            maximum_score = -2
            labels = []
            for index, dic in enumerate(self.stvea.hdbscan_scans):
                score = dic['silhouette_score']
                if score > maximum_score:
                    maximum_score = score
                    maximum_index = index

            self.stvea.codex_cluster = pd.DataFrame(self.stvea.hdbscan_scans[index]['cluster_labels'])

        elif option == 3:
            print('HDBSCAN only')
            # hdbscan only
            data = self.stvea.cite_latent
            reducer = umap.UMAP(n_components=data.shape[1],
                                n_neighbors=15,
                                min_dist=0.1,
                                negative_sample_rate=50,
                                metric=hdbscan_cluster_metric,
                                random_state=hdbscan_random_state)

            umap_latent = reducer.fit_transform(data)

            cluster = hdbscan.HDBSCAN(min_cluster_size=int(hdbscan_min_cluster_size_range),
                                      min_samples=int(hdbscan_min_sample_range),
                                      metric=hdbscan_cluster_metric,
                                      memory='./HDBSCAN_cache')

            hdbscan_labels = cluster.fit_predict(umap_latent)

            self.stvea.cite_cluster = pd.DataFrame(hdbscan_labels)

        if plot_umap:
            self.plot_cite()
