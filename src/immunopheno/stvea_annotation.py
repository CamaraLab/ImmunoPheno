from copy import deepcopy
import re
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import PySimpleGUI as sg
from .stvea_class import STvEA

class Annotation:
    stvea = STvEA()

    def __init__(self, stvea):
        self.stvea = stvea

    def transfer_labels(self, user_input=True):
        """
        This function will show the gene expression levels for each CITE-seq cluster, ask user to input name for each CITE-seq cluster.
        These labels will be transferred to CODEX cells.
        """
        if user_input:
            # show user the gene expression info of the CITE-seq cluster
            cluster_index = self.cluster_heatmap(1)

            # receive user annotation of CITE-seq clusters
            self.cluster_names(cluster_index, 1, option=2)
            cite_cluster_names_dict = self.stvea.cite_cluster_name_dict

        # create indicator matrix of CITE cell cluster assignments
        cite_cluster_assignment = deepcopy(self.stvea.cite_cluster)
        cite_cluster_assignment = cite_cluster_assignment.astype("category")
        cite_cluster_assignment_dummies = pd.get_dummies(cite_cluster_assignment, prefix="", prefix_sep="")
        cite_cluster_assignment_dummies = cite_cluster_assignment_dummies.applymap(lambda x: 1 if x else 0)
        self.stvea.transfer_matrix.columns = cite_cluster_assignment_dummies.index

        # transfer labels from CITE to CODEX
        codex_cluster_names_dummies = self.stvea.transfer_matrix.dot(cite_cluster_assignment_dummies)

        if user_input:
            codex_cluster_index = codex_cluster_names_dummies.apply(lambda row: int(row.idxmax()), axis=1)

            # codex_cluster_names_transferred will store the transferred names.
            # note they are names not cluster indices.
            self.stvea.codex_cluster_names_transferred = pd.DataFrame(
                codex_cluster_index.apply(lambda x: self.stvea.cite_cluster_name_dict.get(int(x))))
        else:
            codex_cluster_index = codex_cluster_names_dummies.apply(lambda row: row.idxmax(), axis=1)
            self.stvea.codex_cluster_names_transferred = codex_cluster_index

    def cluster_heatmap(self,
                        dataset,
                        option=2,
                        upper_limit=10,
                        marker=pd.Series(('Cxcl16', 'Cacnb3', 'Cox6a2',
                                          'Bst2', 'Siglech', 'Hmox1',
                                          'C1qa', 'Vcam1', 'Hba-a1',
                                          'Hbb-bs', 'Hba-a2', 'Ngp',
                                          'S100a9', 'S100a8', 'Klf1',
                                          'Car2', 'Car1', 'Ppt1',
                                          'Ppp1r14a', 'Ffar2', 'Mmp12',
                                          'Jchain', 'Ifi30', 'Vpreb3',
                                          'Wfdc17', 'Ccl9', 'Ccl6',
                                          'Ncr1', 'Ccl5', 'Gzma',
                                          'Cd22', 'Ighd', '1810046K07Rik',
                                          'Zbtb32', 'Bhlhe41', 'Cd3d',
                                          'Cd3e', 'Trbc2'))):
        """
        This function will generate a heatmap of clusters' protein expressions.
        @param marker: a pandas Series that stores marker genes.
        @param dataset: 1 for CITE, 2 for CODEX
        @param option: 1 for static heatmap, 2 for interactive heatmap.
        @return cluster_index
        """
        if dataset == 1:
            # cite
            # cluster assignments for each cell
            clusters = self.stvea.cite_cluster
            # creat a dataframe
            combined_df = deepcopy(self.stvea.cite_mRNA[marker])
            title = "Heatmap of Average CITE-seq Gene Expression by Cluster (mRNA) "
            dataset_type = "CITE-seq"
        else:
            # codex
            clusters = self.stvea.codex_cluster
            # creat a dataframe
            combined_df = deepcopy(self.stvea.codex_protein)
            title = "Heatmap of Average CODEX Gene Expression by Cluster (Protein) "
            dataset_type = "CODEX"


        clusters.index = combined_df.index
        combined_df.insert(len(combined_df.columns), "Cluster", clusters)

        # group based on cluster and calculate the mean for each gene expression level within each cluster
        df_grouped = combined_df.groupby("Cluster").mean()

        # print the size
        print("Size of each cluster for " + dataset_type)
        df_grouped_size = combined_df.groupby("Cluster").size()
        print(df_grouped_size.to_string(header=False))

        if dataset == 1:
            # huge difference between the highest value and lowest value in the df_grouped
            # make it difficult to plot heatplot
            df_grouped = df_grouped.clip(upper=upper_limit)

        # if -1 in df_grouped.index:
        #     # -1 means no cluster assignment
        #     # drop -1
        #     df_grouped.drop(-1, inplace=True)

        # transpose
        df_grouped = df_grouped.transpose()

        if option == 1:
            # generate the static heatmap
            plt.figure(figsize=(10, 10))
            sns.heatmap(df_grouped, cmap="viridis")
            plt.title(title )
            plt.xlabel("Cluster")
            plt.ylabel("Gene")
            plt.show()
        else:
            # generate the interactive heatmap
            # convert the x label to string to avoid some strange behavior of plotly
            x = [str(index) for index in df_grouped.columns]
            fig = go.Figure(data=go.Heatmap(
                z=df_grouped.values,
                x=x,
                y=df_grouped.index,
                hoverongaps=False,
                colorbar={"title": "Average Gene Expression", "titleside": "right"}))

            # configure the graph
            fig.update_layout(title=title + 'Heatmap of Average Gene Expression by Cluster',
                              xaxis_title="Cluster",
                              yaxis_title="Gene",
                              xaxis_tickangle=-45)

            fig.show()

        return df_grouped.columns

    def cluster_names(self, cluster_index, dataset, option=1):
        """
        This function will receive user input for each cluster.
        @param cluster_index: a list contains all the cluster index.
        @param dataset: 1 for CITE, 2 for CODEX
        @return: a dictionary whose key is cluster index, and whose value is its name.
        """
        # let's get the number of clusters
        num_clusters = len(cluster_index)

        if dataset == 1:
            # CITE
            title = " CITE-seq "
        else:
            # CODEX
            title = " CODEX "

        cluster_names = {}
        if option == 1:
        # define layout
            # layout = [[sg.Text('Enter the name for each' + title + 'cluster.')],
            #           *[[sg.Text(cluster_name), sg.Input(key=str(i))]
            #             for i, cluster_name in enumerate(cluster_index)],
            #           [sg.Button('OK')]]

            # # create window
            # window = sg.Window('Enter' + title + 'Cluster Names', layout)

            # # event loop and collecting user input
            # while True:
            #     event, values = window.read()
            #     if event == 'OK':
            #         for i, cluster_index in enumerate(cluster_index):
            #             cluster_names[cluster_index] = values[str(i)].strip()
            #         break
            #     elif event == sg.WINDOW_CLOSED:
            #         break

            # # clean up
            # window.close()
            pass
        else:
            while True:
                user_input = input(f"Complete editing {title} cluster names? y/n\n")
                if user_input == "y" or user_input == "Y":
                    break

            # read txt files
            if dataset == 1:
                file_path = "cite_cluster_names.txt"
            else:
                file_path = "codex_cluster_names.txt"

            with open(file_path, "r") as file:
                for line in file:
                    [name, indices_str] = line.strip().split(": ")
                    indices = indices_str.strip().split(", ")
                    for index in indices:
                        cluster_names[int(index)] = name

            for index in cluster_index:
                if index not in cluster_names.keys():
                    cluster_names[index]=""
                pass

        cluster_names[-1] = ""

        # store the dict in STvEA object
        if dataset == 1:
            # cite
            self.stvea.cite_cluster_name_dict = cluster_names
        else:
            # codex
            self.stvea.codex_cluster_name_dict = cluster_names
        return
