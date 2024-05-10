import numpy as np
import requests
import urllib.parse
import random
import json
import logging
import warnings
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import networkx as nx
from networkx.exception import NetworkXError
from nxontology.imports import from_file
from networkx.algorithms.dag import dag_longest_path

import matplotlib.pyplot as plt
from netgraph import Graph

from sklearn.impute import KNNImputer
from .stvea_controller import Controller

def _update_cl_owl():
    warnings.filterwarnings("ignore")
    response = requests.get('https://www.ebi.ac.uk/ols4/api/ontologies/cl')
    owl_link = response.json()['config']['versionIri']
    return owl_link

def graph_pos(G):
    result = Graph(G, node_layout='dot')
    plt.close()
    return result.node_positions

def find_leaf_nodes(graph, node):
    descendants = nx.descendants(graph, node)
    leaf_nodes = [n for n in descendants if graph.out_degree(n) == 0]
    return leaf_nodes

def subgraph(G: nx.DiGraph, 
             nodes: list, 
             root: str ='CL:0000000')-> nx.DiGraph:
    """
    Constructs a subgraph from a larger graph based on 
    a provided list of leaf nodes

    Considers cell types as the leaf nodes of the subgraph when
    finding the most common ancestor (root node)

    Parameters:
        G (nx.DiGraph): "Master" directed acylic graph containing all 
            possible cell ontology relationships
        nodes (list): cell ontologies of interest - leaf nodes of the graph 
        root (str): root node of the graph

    Returns:
        subgraph (nx.DiGraph): subgraph containing cell ontology relationships
            for only the nodes provided
    """
    if root in nodes:
        raise Exception("Error. Root node cannot be a target node")
    
    node_unions = []
    for node in nodes:
        # For each node in the list, get all of their paths from the root
        node_paths = [set(path) for path in nx.all_simple_paths(G, source=root, target=node)]
        
        if len(node_paths) == 0:
            raise Exception("No paths found. Please enter valid target nodes")
        
        # Then, take the union of those paths to return all possible nodes visited. Store this result in node_unions
        node_union = set.union(*node_paths)
        # Then, add this to a list (node_unions) containing each node's union 
        node_unions.append(node_union)

    # Find a common path by taking the intersection of all unions
    union_inter = set.intersection(*node_unions)

    node_path_lengths = {}
    # Find the distance from each node in union_inter to the root
    for node in union_inter:
        length = nx.shortest_path_length(G, source=root, target=node)
        node_path_lengths[node] = length

    # Get node(s) with the largest path length. This is the lowest common ancestor of [nodes]
    max_value = max(node_path_lengths.values())
    all_LCA = [k for k,v in node_path_lengths.items() if v == max_value]
    
    # Reconstruct a subgraph (DiGraph) from LCA to each node by finding the shortest path
    nodes_union_sps = []
    # for each LCA, reconstruct their graph
    for LCA in all_LCA:
        for node in nodes:
            node_paths = [set(path) for path in nx.all_simple_paths(G, source=LCA, target=node)]
            
            # If target node happens to be a LCA, the node path will be empty
            # skip it
            if len(node_paths) == 0:
                continue

            node_union = set.union(*node_paths)

            # instead, take ALL the paths, and then union of all of those paths
            nodes_union_sps.append(node_union)

    # Take the union of these paths (from each LCA) to return all nodes in the subgraph
    subgraph_nodes = set.union(*nodes_union_sps)
   
    # Create a subgraph
    subgraph = G.subgraph(subgraph_nodes)

    return subgraph

def find_subgraph_from_root(G: nx.DiGraph, 
                            root: str, 
                            leaf_nodes: list):
    all_paths_from_root = []
                                
    # Provided the root node, find all paths to each leaf node
    for leaf in leaf_nodes:
        paths = list(nx.all_simple_paths(G, root, leaf))
        all_paths_from_root.extend(paths)

    # Flatten all paths into a single list of nodes
    all_visited_idCLs = [path for paths in all_paths_from_root for path in paths]

    # Find all unique nodes that have been visited
    unique_visited_idCLs = list(set(all_visited_idCLs))

    # Return subgraph with only those nodes
    subgraph = nx.subgraph(G, unique_visited_idCLs)
    return subgraph
    
def plotly_subgraph(G, nodes_to_highlight, hover_text):
    # Get positions using a layout algorithm from netgraph 'dot'
    pos = graph_pos(G)
    
    # Extract node coordinates for Plotly
    node_x = [pos[node][0] for node in G.nodes]
    node_y = [pos[node][1] for node in G.nodes]
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",  # Include text labels
        hoverinfo="text",
        text=list(G.nodes),  # Use node labels as text
        hovertext=hover_text, # Add in custom hover labels
        line=dict(color="black", width=2)
    )
    
    # Create a list of colors for each node
    node_colors = ["#FCC8C8" if node in nodes_to_highlight else "#C8ECFC" for node in G.nodes]
    
    node_trace.marker = dict(
        color=node_colors,
        size=[20 + 5 * len(str(label)) for label in G.nodes],
        opacity=1,
        line=dict(color="black", width=1),  # Add black circular rim around each node
    )
    
    # Dynamically set node size based on label length
    max_label_length = max(len(str(label)) for label in G.nodes)
    node_trace.marker.size = [12 + 3 * len(str(label)) for label in G.nodes]
    
    # Adjust the font size of the labels
    node_trace.textfont = dict(size=6.5)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    
    # Create layout
    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    # Get depth of graph
    depth = len(dag_longest_path(G))
    adjusted_height = depth * 110
    fig.update_layout(autosize=True,height=adjusted_height)
    
    # Show the plot
    return fig

def convert_idCL_readable(idCL:str):
    idCL_params = {
        'q': idCL,
        'exact': 'true',
        'ontology': 'cl',
        'fieldList': 'label',
        'rows': 1,
        'start': 0
    }

    try:
        res = requests.get("https://www.ebi.ac.uk/ols4/api/search", params=idCL_params)
        res_JSON = res.json()
        cellType = res_JSON['response']['docs'][0]['label']    
        return cellType
    except:
        return ""

def convert_ab_readable(ab_id:str):
    try:
        res = requests.get("http://www.scicrunch.org" + "/resolver/" + ab_id + ".json")
        res_JSON = res.json()
        ab_name = (res_JSON['hits']['hits'][0]
                        ['_source']['antibodies']['primary'][0]
                        ['targets'][0]['name'])
        return ab_name
    except:
        return ""

def plot_antibodies_graph(idCL: str,
                          getAbs_df: pd.DataFrame,
                          plot_df: pd.DataFrame) -> go.Figure():
    idCL_readable = convert_idCL_readable(idCL)
    if idCL_readable == "":
        title = idCL
    else:
        title = f"{idCL_readable} ({idCL})"
    
    antibodies = list(getAbs_df.index) # antibodies to send to endpoint
    
    # Change x axis to use common antibody names
    ab_targets = list(getAbs_df['target'])
    mapping_dict = dict(zip(antibodies, ab_targets))
    # print("mapping_dict:\n", mapping_dict)
    
    modified_mapping_dict = {}
        
    # Add antibody IDs next to their targets in the X-axis
    for i in mapping_dict:
        modified_mapping_dict[i] = str(i) + f" ({mapping_dict[i]})"
    new_ab_targets = list(modified_mapping_dict.values())

    # Rename antibodies to common name
    plot_df['ab_target'] = plot_df['idAntibody'].map(modified_mapping_dict)

    # Use Plotly Express to create a violin plot
    fig = px.violin(plot_df, 
                    x='ab_target', 
                    y=['mean', 'q1', 'median', 'q3', 'min', 'max'], 
                    color='ab_target',
                    category_orders={'ab_target': new_ab_targets},
                    box=True)
    fig.update_layout(title_text=f"Antibodies for: {title}",
                      xaxis_title="Antibody",
                      yaxis_title="Normalized Values",
                              title_x = 0.45, 
                              font=dict(size=8),
                              autosize=True,
                              height=600)

    fig.update_traces(width=0.75, selector=dict(type='violin'))
    fig.update_traces(marker={'size': 1})

    return fig

def plot_celltypes_graph(ab_id: str,
                         getct_df: pd.DataFrame,
                         plot_df: pd.DataFrame) -> go.Figure():
    ab_readable = convert_ab_readable(ab_id)
    if ab_readable == "":
        title = ab_id
    else:
        title = f"{ab_readable} ({ab_id})"
    
    celltypes = list(getct_df.index) # antibodies to send to endpoint
    
    # Change x axis to use common antibody names
    celltypes_names = list(getct_df['cellType'])
    mapping_dict = dict(zip(celltypes, celltypes_names))
    
    modified_mapping_dict = {}
        
    # Add cell type ids (idCL) next to their name in the X-axis
    for i in mapping_dict:
        modified_mapping_dict[i] = str(i) + f" ({mapping_dict[i]})"
    new_celltypes = list(modified_mapping_dict.values())

    # Rename antibodies to common name
    plot_df['celltype'] = plot_df['idCL'].map(modified_mapping_dict)

    # Use Plotly Express to create a violin plot
    fig = px.violin(plot_df, 
                    x='celltype', 
                    y=['mean', 'q1', 'median', 'q3', 'min', 'max'], 
                    color='celltype',
                    category_orders={'celltype': new_celltypes},
                    box=True)
    fig.update_layout(title_text=f"Cell Types for: {title}",
                      xaxis_title="Cell Type",
                      yaxis_title="Normalized Values",
                              title_x = 0.45, 
                              font=dict(size=8),
                              autosize=True,
                              height=600)

    fig.update_traces(width=0.75, selector=dict(type='violin'))
    fig.update_traces(marker={'size': 1})

    return fig

def remove_all_zeros_or_na(protein_df):
    # Check if any row in the DataFrame has all NAs or all zeros
    rows_to_exclude = protein_df.apply(lambda row: all(row.isna() | (row == 0)), axis=1)
    
    # Filter out rows to keep only those that do not meet the exclusion conditions
    filtered_df_rows = protein_df[~rows_to_exclude]

    # Check if any column in the DataFrame has all NAs or all zeros
    columns_to_exclude = filtered_df_rows.apply(lambda col: all(col.isna() | (col == 0)), axis=0)
    
    # Filter out columns to keep only those that do not meet the exclusion conditions
    filtered_df = filtered_df_rows.loc[:, ~columns_to_exclude]

    # Display the modified DataFrame
    return filtered_df

def filter_imputed(imputed_df_with_na, rho):
    # First check if dataframe has any NAs to start with
    has_na = imputed_df_with_na.isna().any().any()
    if not has_na:
        return imputed_df_with_na

    # Extract all columns other than "idCL"
    imputed_ab = imputed_df_with_na.loc[:, imputed_df_with_na.columns != "idCL"]

    # Create a "mask" table where 0s = values not NA in the imputed table, 1s = values that were NAs in the imputed table
    imputed_ab_mask = imputed_ab.isna()
    imputed_bool = imputed_ab_mask.astype(int)

    ## Creating weights for ROWS
    # For each row, count the number of 1s and sum them
    row_sums = imputed_bool.sum(axis=1)
    
    # Create a new DataFrame to store row index and its sum
    row_sums_df = pd.DataFrame({'Row': imputed_bool.index, 'Sum': row_sums})

    # Apply weights based on the number of rows
    row_sums_df['Weighted_Sum'] = row_sums_df['Sum'] * len(imputed_bool.index) * rho

    ## Creating weights for COLUMNS
    # For each column, count the number of 1s and sum them
    col_sums = imputed_bool.sum(axis=0)
    
    # Create a new DataFrame to store column index and its sum
    col_sums_df = pd.DataFrame({'Column': imputed_bool.columns, 'Sum': col_sums})
    
    # Apply weights based on the number of columns
    col_sums_df['Weighted_Sum'] = col_sums_df['Sum'] * len(imputed_bool.columns) * (1 - rho)

    ## Finding max weighted sum & frequency for ROWS
    max_row_sum = row_sums_df['Weighted_Sum'].max()
    
    # Find rows with the max sum
    max_row_sum_rows = row_sums_df[row_sums_df['Weighted_Sum'] == max_row_sum]
    
    # Check if there are multiple rows with the max sum
    multiple_max_row_sums = len(max_row_sum_rows) > 1
    
    ## Finding max weighted sum & frequency for COLUMNS
    # Find the max sum in columns
    max_col_sum = col_sums_df['Weighted_Sum'].max()
    
    # Find columns with the max sum
    max_col_sum_cols = col_sums_df[col_sums_df['Weighted_Sum'] == max_col_sum]
    
    # Check if there are multiple columns with the max sum
    multiple_max_col_sums = len(max_col_sum_cols) > 1

    # Deciding which row or column to filter out
    # First priority: higher max weighted sum
    if max_row_sum > max_col_sum:
        # In this case, we want to filter out all the rows with this max sum
        filtered_imputed = imputed_df_with_na.loc[~imputed_df_with_na.index.isin(max_row_sum_rows.index)]
        
    elif max_col_sum > max_row_sum:
        # In this case, we want to filter out all columns with this max sum
        filtered_imputed = imputed_df_with_na.drop(columns=list(max_col_sum_cols.index))
        
    elif max_row_sum == max_row_sum:
        # Tiebreaker condition based on frequency of the max weight
        if len(max_row_sum_rows) > len(max_col_sum_cols):
            # If there were more frequent max sums in rows, then drop those
            filtered_imputed = imputed_df_with_na.loc[~imputed_df_with_na.index.isin(max_row_sum_rows.index)]
        elif len(max_col_sum_cols) > len(max_row_sum_rows):
            # If there were more frequent max sums in cols, then drop those
            filtered_imputed = imputed_df_with_na.drop(columns=list(max_col_sum_cols.index))

        elif len(max_row_sum_rows) == len(max_col_sum_cols):
            # If there is the same max, and the same frequency in either row/columns, randomly choose
            # Generate a random number between 0 and 1
            random_number = random.randint(0, 1)

            if random_number == 0: # Drop rows
                filtered_imputed = imputed_df_with_na.loc[~imputed_df_with_na.index.isin(max_row_sum_rows.index)]
            elif random_number == 1: # Drop columns
                filtered_imputed = imputed_df_with_na.drop(columns=list(max_col_sum_cols.index))
            
    return filtered_imputed

def keep_calling_filter_imputed(original_imputed_df, rho):
    # Empty dataframe to hold results & constantly update
    modified_imputed_df = pd.DataFrame()
    
    # Initialize a flag to track changes
    still_has_NA = True
    
    # Keep looping until no NAs remain in the dataframe
    while still_has_NA:
        # Call filter_imputed and get the modified imputed table
        modified_imputed_df = filter_imputed(original_imputed_df, rho=rho)
        
        # Check if the current modified_imputed_df has any NAs remaining
        if not (modified_imputed_df.isna().any().any()):
            # If they are the same, set the flag to False to exit the loop
            still_has_NA = False
        else:
            # If there are still NAs, update the imputed_df with the new modified_imputed_df for next call
            original_imputed_df = modified_imputed_df
    
    # Return the final modified_idCLs
    return modified_imputed_df

def impute_dataset_by_type(downsampled_df, rho):
    # Find all unique idCLs in the table
    unique_idCLs = list(set(downsampled_df['idCL']))

    imputed_dataframes = []
    
    # For each idCL, find the rows in the table
    for idCL in unique_idCLs:
        subtable = downsampled_df.loc[downsampled_df['idCL'] == idCL]

        # Get all antibody values from this table (exclude last two columns). This is what will be imputed
        remaining_ab = [ab for ab in subtable.columns if (ab != 'idCL' and ab != 'idExperiment')]
        subtable_ab_to_impute = subtable[remaining_ab]
        # Handle cases where a column (antibody) is all NaN
        subtable_ab_drop = subtable_ab_to_impute.dropna(axis="columns", how="all")

        # Dynamically adjust k. If num cells in table < 10, set k = num cells
        # Otherwise, k will be 10 by default
        if (len(subtable_ab_drop)) < 10:
            k = len(subtable_ab_drop)
        else:
            k = 10

        # Impute the values in
        imputer = KNNImputer(n_neighbors=k, weights="distance")
        imputed_np = imputer.fit_transform(subtable_ab_drop.to_numpy())
        
        # Put these imputed values back into the dataframe
        imputed_df = pd.DataFrame(imputed_np, index=subtable_ab_drop.index, columns=subtable_ab_drop.columns)
        imputed_dataframes.append(imputed_df)

    # Combine all imputed dataframes back to each other by row
    combined_imputed_df = pd.concat(imputed_dataframes, axis=0)

    # For antibodies that still have NAs after imputation, repeatedly filter them out based on row/column heuristic
    # combined_imputed_dropped_df = combined_imputed_df.dropna(axis="columns", how="any")
    combined_imputed_dropped_df = keep_calling_filter_imputed(combined_imputed_df, rho=rho)
    
    # Retrieve all the idCLs again for all the cells
    combined_imputed_dropped_idCLs = downsampled_df.loc[combined_imputed_dropped_df.index]["idCL"]
    
    combined_imputed_df_with_idCL = pd.concat([combined_imputed_dropped_df, combined_imputed_dropped_idCLs], axis=1)

    # Compare final output with the original output. See what remains, and see whether they were orignally NAs
    final_columns = combined_imputed_df_with_idCL.columns
    final_index = combined_imputed_df_with_idCL.index
    original_remains = downsampled_df.loc[final_index, final_columns]

    # Find statistics on the number of antibodies, cells, cell types that were imputed
    num_columns_with_na = original_remains.isna().any().sum()
    print("Number of antibodies imputed:", num_columns_with_na)

    num_rows_with_na = original_remains.isna().any(axis=1).sum()
    print("Number of cells imputed:", num_rows_with_na)

    # Find which rows (cells) were NAs. From those cells, find number of unique cell types
    na_rows = original_remains[original_remains.isna().any(axis=1)]
    print("Number of cell types imputed:", len(set(na_rows["idCL"])))
    
    return combined_imputed_df_with_idCL

def convert_idCL_readable(idCL:str) -> str:
    """
    Converts a cell ontology id (CL:XXXXXXX) into a readable cell type name

    Parameters:
        idCL (str): cell ontology ID

    Returns:
        cellType (str): readable cell type name
        
    """
    idCL_params = {
        'q': idCL,
        'exact': 'true',
        'ontology': 'cl',
        'fieldList': 'label',
        'rows': 1,
        'start': 0
    }

    try:
        res = requests.get("https://www.ebi.ac.uk/ols4/api/search", params=idCL_params)
        res_JSON = res.json()
        cellType = res_JSON['response']['docs'][0]['label']
    except:
        cellType = idCL
    
    return cellType

def ebi_idCL_map(labels_df: pd.DataFrame) -> dict:
    """
    Converts a list of cell ontology IDs into readable cell type names
    as a dictionary

    Parameters:
        labels_df (pd.DataFrame): dataframe with cell labels from singleR
    
    Returns:
        idCL_map (dict) : dictionary mapping cell ontology ID to cell type
    
    """
    idCL_map = {}
    
    idCLs = set(labels_df["labels"])
    
    for idCL in idCLs:
        idCL_map[idCL] = convert_idCL_readable(idCL)
    
    return idCL_map

class ImmunoPhenoDB_Connect:
    def __init__(self, url: str):
        self.url = url
        self._OWL_graph = None
        self._subgraph = None
        self._db_idCLs = None
        self._db_idCL_names = None
        self._last_stvea_params = None
        self.imputed_reference = None
        self.transfer_matrix = None

        if self.url is None:
            raise Exception("Error. Server URL must be provided")

        if self.url is not None and self.url.endswith("/"):
            # Find the last forward slash
            last_slash_index = self.url.rfind("/")
            
            # Remove everything after the last forward slash
            result_url = self.url[:last_slash_index]
            self.url = result_url
        
        if "://" not in self.url:
            self.url = "http://" + self.url
        
        if self._OWL_graph is None:
            print("Loading necessary files...")
            owl_link = _update_cl_owl()
            G_nxo = from_file(owl_link)
            G = G_nxo.graph
            self._OWL_graph = G
        
        if self._subgraph is None:
            # Make an API call to get our unique idCLs
            print("Connecting to database...")
            try:
                idCL_response = requests.get(f"{self.url}/api/idcls")
                idCL_JSON = idCL_response.json()
                idCLs = idCL_JSON['idCLs']

                self._db_idCLs = idCLs
                self._subgraph = subgraph(self._OWL_graph, self._db_idCLs)

                convert_idCL = {
                    "idCL": list(self._subgraph.nodes)
                }

                convert_idCL_res = requests.post(f"{self.url}/api/convertcelltype", json=convert_idCL)
                idCL_names = convert_idCL_res.json()["results"]                
                self._db_idCL_names = idCL_names
                
                print("Connected to database.")
            except:
                raise Exception("Error. Unable to connect to database")
    
    def _find_descendants(self, id_CLs: list) -> dict:
        node_fam_dict = {}
    
        # For each idCL, find all of their unique descendants using the database graph
        for idCL in id_CLs:
            node_family = []
            descendants = nx.descendants(self._subgraph, idCL)
            node_family.extend(list(set(descendants)))
            node_fam_dict[idCL] = node_family

        return node_fam_dict
    
    def plot_db_graph(self, root=None):
        if root is None:
            # We already calculated the database's subgraph in self._subgraph
            # Find hover names
            hover_names = []
            for node in list(self._subgraph.nodes):
                hover_names.append(self._db_idCL_names[node])
            plotly_graph = plotly_subgraph(self._subgraph, self._db_idCLs, hover_names)
        else:
            leaf_nodes = find_leaf_nodes(self._subgraph, root)
            
            if len(leaf_nodes) == 0:
                # If there are no leaf nodes of root, then we were already provided a leaf node
                # We can plot this singular node directly
                nodes_to_plot = [root]
                # Check if this node was in our database
                node_in_db = list(set(nodes_to_plot) & set(self._db_idCLs))
                # Take subgraph using default function
                default_subgraph = nx.subgraph(self._subgraph, nodes_to_plot)
                # Find hover names
                hover_names = []
                for node in list(default_subgraph.nodes):
                    hover_names.append(self._db_idCL_names[node])
                plotly_graph = plotly_subgraph(default_subgraph, node_in_db, hover_names)
                
            elif len(leaf_nodes) == 1:
                # If there was only one leaf node, we can directly plot the descendants to that node
                nodes_to_plot = list(nx.descendants(self._subgraph, root))
                # Include the original node
                nodes_to_plot.insert(0, root)
                # Check if these were in the database
                node_in_db = list(set(nodes_to_plot) & set(self._db_idCLs))
                # Take subgraph using default function
                default_subgraph = nx.subgraph(self._subgraph, nodes_to_plot)
                # Find hover names
                hover_names = []
                for node in list(default_subgraph.nodes):
                    hover_names.append(self._db_idCL_names[node])
                plotly_graph = plotly_subgraph(default_subgraph, node_in_db, hover_names)
                
            else:
                # Multiple leaf nodes require finding the lowest common ancestor
                # Use custom subgraph function
                custom_subgraph = find_subgraph_from_root(self._subgraph, root, leaf_nodes)
                # Include the original node
                nodes_to_plot = list(custom_subgraph.nodes)
                # Check if these were in the database
                node_in_db = list(set(nodes_to_plot) & set(self._db_idCLs))
                # Find hover names
                hover_names = []
                for node in list(custom_subgraph.nodes):
                    hover_names.append(self._db_idCL_names[node])
                plotly_graph = plotly_subgraph(custom_subgraph, node_in_db, hover_names)
                
        return plotly_graph

    def find_antibodies(self, 
                        id_CLs: list,
                        background_id_CLs: list = None,
                        idBTO: list = None, 
                        idExperiment: list = None)-> tuple: 
        
        # First find all descendants of the provided id_CLs. These will be included 
        # when running the LMM
        try: 
            node_fam_dict = self._find_descendants(id_CLs)
            if background_id_CLs is not None:
                background_fam_dict = self._find_descendants(background_id_CLs)
            else:
                background_fam_dict = None
        except NetworkXError as err:
            err_msg = str(err).split(' ')[2] # Get idCL error
            raise Exception(f"Error. {err_msg} not found in the database")
        
        # Call API endpoint here to get_Abs to return dataframe        
        abs_body = {
            "idCL": node_fam_dict,
            "background": background_fam_dict,
            "idBTO": idBTO,
            "idExperiment": idExperiment
        }
                            
        abs_response = requests.post(f"{self.url}/api/findabs", json=abs_body)

        # Check response from server
        if 'text/html' in abs_response.headers.get('content-type'):
            error_msg = abs_response.text
            raise ValueError("No antibodies found. Please retry with fewer cell types or different parameters.")
        elif 'application/json' in abs_response.headers.get('content-type'):
            res_JSON = abs_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)

        # Store all summary plots for each idCL here
        idCL_plots = {}
                            
        # Antibodies to send to plot_antibodies endpoint
        antibodies = list(res_df.index) 

        # Store all figures in a dict (key: idCL, value: figure)
        all_figures = {}

        for idCL_parent, descendants in node_fam_dict.items():
            desc_copy = descendants.copy()
            desc_copy.insert(0, idCL_parent) # include parent when sending all idCLs to endpoint

            plot_abs_body = {
                "abs": antibodies,
                "idcls": desc_copy
            }

            abs_plot_response = requests.post(f"{self.url}/api/plotabs", json=plot_abs_body) # send antibodies and idCLs
            abs_plot_JSON = abs_plot_response.json() # returns a dictionary
            abs_plot_df = pd.DataFrame.from_dict(abs_plot_JSON) # Convert into a dataframe
            
            # Add to dictionary
            idCL_plots[idCL_parent] = abs_plot_df

        # Plot using plotly
        for idCL, summary_stats_df in idCL_plots.items():
            fig = plot_antibodies_graph(idCL, res_df, summary_stats_df)
            all_figures[idCL] = fig
    
        return (res_df, all_figures)
    
    def find_celltypes(self,
                       ab_ids: list,
                       idBTO: list = None, 
                       idExperiment: list = None) -> tuple: 
        # Dict to hold results (dataframe) for each antibody
        ab_df_dict = {}
        
        celltypes_body = {
            "ab": ab_ids,
            "idBTO": idBTO,
            "idExp": idExperiment
        }
        
        # Call API endpoint here to get dataframe
        celltypes_response = requests.post(f"{self.url}/api/findcelltypes", json=celltypes_body)
                           
        if 'text/html' in celltypes_response.headers.get('content-type'):
            res_dict_strjson = celltypes_response.text
            raise ValueError("No cell types found. Please try with different parameters.")
        elif 'application/json' in celltypes_response.headers.get('content-type'):
            res_dict_strjson = celltypes_response.json() # returns a dict of string-jsons
        
        # Convert all of these string-jsons into dataframes
        for key, value in res_dict_strjson.items():
            # Convert string-json into dict
            temp_dict = json.loads(value)
            
            # Convert dict into dataframe
            temp_df = pd.DataFrame.from_dict(temp_dict)
            
            # Add to final dict of dfs
            ab_df_dict[key] = temp_df
            
        # Check for skipped antibodies If so, then the server found no cells in the database
        res_abs = list(res_dict_strjson.keys())
        missing_abs = list(set(ab_ids) - set(res_abs))
        for missing in missing_abs:
            logging.warning(f"No cells found in the database for {missing}. Skipping {missing}")
            
        plotting_dfs = {} # key: ab, value: df
        
        for ab, celltype_res_df in ab_df_dict.items():
            # Extract index from each dataframe
            df_idCLs = list(celltype_res_df.index)
            
            # Send antibody and idCLs to get dataframe
            plot_celltypes_body = {
                "ab": ab,
                "idcls": df_idCLs
            }
            
            celltypes_plot_response = requests.post(f"{self.url}/api/plotcelltypes", json=plot_celltypes_body)
            celltypes_plot_JSON = celltypes_plot_response.json() # returns a dict
            celltypes_plot_df = pd.DataFrame.from_dict(celltypes_plot_JSON) # Convert into a dataframe

            # Store all of these in a dict
            plotting_dfs[ab] = celltypes_plot_df

        # Plot here using plotly
        all_figures = {}
        for ab, celltypes_plot_df in plotting_dfs.items():
            fig = plot_celltypes_graph(ab, ab_df_dict[ab], celltypes_plot_df)
            all_figures[ab] = fig
            
        return ab_df_dict, all_figures
    
    def find_experiments(self,
                         ab: list = None,
                         idCL: list = None,
                         idBTO: list = None) -> pd.DataFrame:
        exp_body = {
            "ab": ab,
            "idCL": idCL,
            "idBTO": idBTO
        }
        
        exp_response = requests.post(f"{self.url}/api/findexperiments", json=exp_body)
        if 'text/html' in exp_response.headers.get('content-type'):
            return exp_response.text
        elif 'application/json' in exp_response.headers.get('content-type'):
            res_JSON = exp_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
            return res_df
    
    def which_antibodies(self,
                         search_query: str) -> pd.DataFrame:
        
        wa_body = {
            "search_query": search_query
        }
        
        wa_response = requests.post(f"{self.url}/api/whichantibodies", json=wa_body)
        if 'text/html' in wa_response.headers.get('content-type'):
            return wa_response.text
        elif 'application/json' in wa_response.headers.get('content-type'):
            res_JSON = wa_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
            return res_df
    
    def which_celltypes(self,
                        search_query: str) -> pd.DataFrame:
        
        wc_body = {
            "search_query": search_query
        }
        
        wc_response = requests.post(f"{self.url}/api/whichcelltypes", json=wc_body)
        if 'text/html' in wc_response.headers.get('content-type'):
            return wc_response.text
        elif 'application/json' in wc_response.headers.get('content-type'):
            res_JSON = wc_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
            return res_df
    
    def which_experiments(self,
                          search_query: str) -> pd.DataFrame:
        we_body = {
            "search_query": search_query
        }
        we_response = requests.post(f"{self.url}/api/whichexperiments", json=we_body)
        if 'text/html' in we_response.headers.get('content-type'):
            return we_response.text
        elif 'application/json' in we_response.headers.get('content-type'):
            res_JSON = we_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
            return res_df

    def run_stvea(self,
                  IPD,
                  idBTO: list = None, 
                  idExperiment: list = None, 
                  parse_option: int = 1,
                  rho: float = 0.5,
                  pairwise_threshold: float = 1.0, 
                  na_threshold: float = 1.0, 
                  population_size: int = 50,
                  # STvEA parameters
                  k_find_nn: int = 80,
                  k_find_anchor: int = 20,
                  k_filter_anchor: int = 100,
                  k_score_anchor: int = 80,
                  k_find_weights: int = 100,
                  k_transfer_matrix = None,
                  c_transfer_matrix: float = 0.1,
                  mask_threshold: float = 0.5,
                  mask: bool = True,
                  num_chunks=1,
                  num_cores=1):

        # Check if reference query parameters have changed OR if the reference table is empty
        if (IPD, IPD._stvea_correction_value, idBTO, idExperiment, 
            parse_option, rho, pairwise_threshold, 
            na_threshold, population_size) != self._last_stvea_params or self.imputed_reference is None:

            antibody_pairs = [[key, value] for key, value in IPD._ab_ids_dict.items()]
        
            stvea_body = {
                "antibody_pairs": antibody_pairs,
                "idBTO": idBTO,
                "idExperiment": idExperiment,
                "parse_option": parse_option,
                "pairwise_threshold": pairwise_threshold,
                "na_threshold": na_threshold,
                "population_size": population_size
            }

            print("Retrieving reference dataset...")
            stvea_response = requests.post(f"{self.url}/api/stveareference", json=stvea_body)
            if 'text/html' in stvea_response.headers.get('content-type'):
                return stvea_response.text
            elif 'application/json' in stvea_response.headers.get('content-type'):
                res_JSON = stvea_response.json()
                reference_dataset = pd.DataFrame.from_dict(res_JSON)

            # Output statistics on the number of antibodies matched
            columns_to_exclude = ["idCL", "idExperiment"]
            num_antibodies_matched = (~reference_dataset.columns.isin(columns_to_exclude)).sum()
            if parse_option == 1:
                print(f"Number of antibodies matched from database using clone ID: {num_antibodies_matched}")
            elif parse_option == 2:
                print(f"Number of antibodies matched from database using antibody target: {num_antibodies_matched}")
            elif parse_option == 3:
                print(f"Number of antibodies matched from database using antibody ID: {num_antibodies_matched}")

            # Impute any missing values in reference dataset
            print("Imputing missing values...")
            imputed_reference = impute_dataset_by_type(reference_dataset, rho=rho) 

            # Apply stvea_correction value
            self.imputed_reference = imputed_reference.copy(deep=True).applymap(
                        lambda x: x - IPD._stvea_correction_value if (x != 0 and type(x) is not str) else x)

            # Store these parameters to check for subsequent function calls
            self._last_stvea_params = (IPD, IPD._stvea_correction_value, idBTO, idExperiment, 
                                       parse_option, rho, pairwise_threshold, 
                                       na_threshold, population_size)

        # Separate out the antibody counts from the cell IDs 
        imputed_antibodies = self.imputed_reference.loc[:, self.imputed_reference.columns != 'idCL']
        imputed_idCLs = self.imputed_reference['idCL'].to_frame()
    
        # Convert antibody names from CODEX normalized counts to their IDs
        codex_normalized_with_ids = IPD._normalized_counts_df.rename(columns=IPD._ab_ids_dict, inplace=False)
    
        # Perform check for rows/columns with all 0s or NAs
        codex_normalized_with_ids = remove_all_zeros_or_na(codex_normalized_with_ids)
                        
        # At this stage, we have all the information we need to run STvEA
        print("Running STvEA...")
        cn = Controller()
        cn.interface(codex_protein=codex_normalized_with_ids, 
                     cite_protein=imputed_antibodies,
                     cite_cluster=imputed_idCLs,
                     k_find_nn=k_find_nn,
                     k_find_anchor=k_find_anchor,
                     k_filter_anchor=k_filter_anchor,
                     k_score_anchor=k_score_anchor,
                     k_find_weights=k_find_weights,
                     # transfer_matrix
                     k_transfer_matrix=k_transfer_matrix,
                     c_transfer_matrix=c_transfer_matrix,
                     mask_threshold=mask_threshold,
                     mask=mask,
                     num_chunks=num_chunks,
                     num_cores=num_cores)

        # Store transfer_matrix in class
        transfer_matrix = cn.stvea.transfer_matrix
        self.transfer_matrix = transfer_matrix
        
        transferred_labels = cn.stvea.codex_cluster_names_transferred
        
        # Add the labels to the IPD object
        labels_df = transferred_labels.to_frame(name="labels")
        convert_idCL = {
            "idCL": list(set(labels_df['labels']))
        }
        convert_idCL_res = requests.post(f"{self.url}/api/convertcelltype", json=convert_idCL)
        idCL_names = convert_idCL_res.json()["results"]     
        labels_df['celltype'] = labels_df['labels'].map(idCL_names)
        
        # Before setting norm_cell_types, check if it matches the previous. If not, reset norm_umap field
        if not (labels_df.equals(IPD._cell_labels_filt_df)):
            IPD._norm_umap = None
        IPD._cell_labels_filt_df = labels_df
        
        # Add labels to raw cell labels as well. The filtered rows will be marked as "filtered"
        original_cells_index = IPD.protein.index
        merged_df = IPD._cell_labels_filt_df.reindex(original_cells_index)
        merged_df = merged_df.fillna("filtered")

        # Check if the raw and norm labels have changed. If so, reset the UMAP field in IPD
        if not (merged_df.equals(IPD._cell_labels)):
            IPD._raw_umap = None
        IPD._cell_labels = merged_df

        # Make sure the indexes match
        IPD._normalized_counts_df = IPD._normalized_counts_df.loc[IPD._cell_labels_filt_df.index]
        print("Annotation transfer complete.")
        return IPD