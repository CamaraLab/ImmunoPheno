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
import math
import scipy
import copy
from importlib.resources import files
from scipy.stats import entropy

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

# Functions used for run_stvea
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
    
    # Remove any rows/columns that are all 0s
    combined_imputed_dropped_filtered_df = remove_all_zeros_or_na(combined_imputed_dropped_df)

    # Retrieve all the idCLs again for all the cells
    combined_imputed_dropped_idCLs = downsampled_df.loc[combined_imputed_dropped_filtered_df.index]["idCL"]
    combined_imputed_df_with_idCL = pd.concat([combined_imputed_dropped_filtered_df, combined_imputed_dropped_idCLs], axis=1)

    # Compare final output with the original output. See what remains, and see whether they were orignally NAs
    final_columns = combined_imputed_df_with_idCL.columns
    final_index = combined_imputed_df_with_idCL.index
    original_remains = downsampled_df.loc[final_index, final_columns]

    # Find statistics on the number of antibodies, cells, cell types that were imputed
    num_columns_with_na = original_remains.isna().any().sum()
    print("Number of antibodies imputed:", num_columns_with_na)

    num_rows_with_na = original_remains.isna().any(axis=1).sum()
    print("Total number of cells returned:", len(final_index))
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

# Functions used for filter_labels
def downsample(entire_reference_table: pd.DataFrame,
               downsample_size: int = 10000,
               size: int = 50) -> pd.DataFrame:
    """
    Downsamples the large reference table to 10,000 rows (cells). 
    Performs downsampling in a controlled randomized order, where
    proportions of each cell type in the original table are 
    kept in the downsampled table. 

    Parameters:
        entire_reference_table (pd.DataFrame): original large table
            containing all cells for the given antibodies
        size (int): the minimum number of cells needed to define 
            a cell type population
    
    Returns:
        entire_reference_table (pd.DataFrame): downsampled 
            reference table
    """

    total_num_cells = len(entire_reference_table.index)
    
    # Downsample if number of rows exceeds 10,000
    if total_num_cells <= downsample_size:
        return entire_reference_table
    else:
        cells_to_keep = []
        combined_dfs = []
        
        # Find all unique idCLs in the table
        unique_idCLs = list(set(entire_reference_table['idCL']))

        # For each idCL, calculate the number of cells present
        for idCL in unique_idCLs:
            idCL_cells = entire_reference_table.loc[entire_reference_table['idCL'] == idCL]

            # Find number of cells for this cell type
            list_of_cells = list(idCL_cells.index)
            num_idCL_cells = len(list_of_cells)

            # Calculate an adjusted sample_amount for each idCL population to choose from
            sample_amount = ((num_idCL_cells)/(total_num_cells)) * downsample_size

            # Round up the sample amount
            sample_amount_rounded_up = math.ceil(sample_amount)
            
            if sample_amount_rounded_up > size:
                # Randomly sample this number of cells from this idCL population
                sampled_population_index = random.sample(list_of_cells, sample_amount_rounded_up)

                # Add these cells to cells_to_keep
                cells_to_keep.extend(sampled_population_index)

                # Create a df for just these cells in this cell type
                temp_df = idCL_cells.loc[sampled_population_index]

                # Add this to combined_dfs
                combined_dfs.append(temp_df)
                
            else:
                # If the sample amount was below our threshold, take 50 of the cells remaining
                if num_idCL_cells > size:
                    # Take 50 of these cells
                    smaller_sampled_population_index = random.sample(list_of_cells, size)

                    # Add these cells to cells_to_keep
                    cells_to_keep.extend(smaller_sampled_population_index)

                    # Create a df for just these cells in this cell type
                    temp_df = idCL_cells.loc[smaller_sampled_population_index]
                    
                    # Add this to combined_dfs
                    combined_dfs.append(temp_df)
                    
                # If there is not even 50 cells in the population, take whatever remains
                else:
                    remaining_sample_population_index = list_of_cells
                    
                    # Add these cells to cells_to_keep
                    cells_to_keep.extend(remaining_sample_population_index)

                    # Create a df for just these cells in this cell type
                    temp_df = idCL_cells.loc[remaining_sample_population_index]
                    
                    # Add this to combined_dfs
                    combined_dfs.append(temp_df)

        if len(cells_to_keep) > downsample_size:
            reduced_cells_to_keep = random.sample(cells_to_keep, downsample_size)
            return entire_reference_table.loc[pd.Index(reduced_cells_to_keep)]
        else:
            return entire_reference_table.loc[pd.Index(cells_to_keep)]

def pearson_correlation_adjaceny_matrix(pairwise_pearson_correlation_distances):
    # Find the median correlation distance in the entire distance matrix
    median = pairwise_pearson_correlation_distances.stack().median()
    
    # Create a boolean mask for values under a threshold
    # We will only consider pairs that fall under this threshold
    filtered_bool_mask = pd.DataFrame(pairwise_pearson_correlation_distances) < median

    # Create adjacency matrix where 0 = false, 1 = true
    adjacency_matrix = filtered_bool_mask.astype(int)
    return adjacency_matrix

def fast_cell_label_graph(adj_mat):
    # Convert adjacency matrix to sparse and create graph
    G = nx.from_scipy_sparse_array(scipy.sparse.csr_matrix(adj_mat))

    node_names = list(adj_mat.index)
    node_name_map = {i: node_names[i] for i in range(len(node_names))}

    # Rename nodes in graph
    G = nx.relabel_nodes(G, node_name_map)

    return G

#   Part 1 Filtering
def run_fisher_exact_test(G, celltype_of_interest):
    # Initialize counts for each quadrant of the contingency table
    quadrant1_count = 0  # Both nodes do not have "celltype"
    quadrant2_count = 0  # Both nodes have "celltype"
    quadrant3_count = 0  # One node has "celltype" and the other does not

    # Iterate over the edges in the graph
    for u, v in G.edges:
        # Check the attributes of the nodes at each end of the edge
        u_cell_type = G.nodes[u]["celltype"]
        v_cell_type = G.nodes[v]["celltype"]
        
        # Case 1: Both nodes do not have "celltype1"
        if u_cell_type != celltype_of_interest and v_cell_type != celltype_of_interest:
            quadrant1_count += 1
            
        # Case 2: Both nodes have "celltype1"
        elif u_cell_type == celltype_of_interest and v_cell_type == celltype_of_interest:
            quadrant2_count += 1
            
        # Case 3: One node has "celltype1" and the other does not
        else:
            quadrant3_count += 1

    contingency_table = [
        [quadrant1_count, quadrant3_count],
        [quadrant3_count, quadrant2_count]
    ]

    p_value = scipy.stats.fisher_exact(contingency_table)
    return p_value[1] 

def ab_id_dict_from_df(cell_labels_df, idCL_column="labels", celltype_column="celltype"):
    unique_pairs_df = cell_labels_df.drop_duplicates()
    result_dict = dict(zip(unique_pairs_df[idCL_column], unique_pairs_df[celltype_column]))
    return result_dict

def compare_idCL(G, cl1, cl2):
    # Make sure graph is undirected
    G_undirected = G.to_undirected()
    
    # Find distance between the two idCL nodes
    distance = nx.shortest_path_length(G_undirected, source=cl1, target=cl2)
    return distance

#   Part 2 Filtering
def part2_filter(owl_graph, downsample_graph, idCLs, cell_labels_df, p_threshold=0.05, epsilon=4) -> list:
    """
    This function needs to take in 2 different graphs
    1. OWL graph: UNDIRECTED Original graph containing all cell type relationships. Used to calculate distances between nodes
    2. Downsample Graph: Graph we generate from the normalized cell labels. Used to generate subgraph and run the fisher exact test

    This takes in and returns a modified list of idCLs to keep performing pairwise comparisons
    1. idCLs: can contain names that's either a single idCL/celltype OR a combination of CL1_CL2

    This takes in a dataframe containing the cell labels/celltype names, which will be modified as well
    1. cell_labels_df

    P_threshold is for the fisher exact test
    epsilon is for satisfying one of 3 conditions after receiving a insignificant P value
    """   
    # For finding distances between two cell type nodes, use an undirected OWL graph which only contains CL nodes
    # Create a mapping dictionary to quickly find the celltype name for an ID
    ab_lookup_dict = ab_id_dict_from_df(cell_labels_df)

    # Return the modified list of idCLs at the end
    modified_idCLs = idCLs.copy()

    # Begin going through every possible pair of cell types
    for i in range(len(idCLs)):
        for j in range(i+1, len(idCLs)):
            distance_between_nodes = compare_idCL(owl_graph, idCLs[i], idCLs[j])

            # If the distance between two nodes (cell types) is <=2 , create a NN graph
            if distance_between_nodes <= 2:
   
                # Use the DOWNSAMPLED graph we made here, we need the directed edges
                selected_nodes_cl1 = [n for n,v in downsample_graph.nodes(data=True) if v['celltype'] == idCLs[i]]
                selected_nodes_cl2 = [n for n,v in downsample_graph.nodes(data=True) if v['celltype'] == idCLs[j]]
                selected_nodes_total = selected_nodes_cl1 + selected_nodes_cl2

                # Get the sub-graph of the downsampled graph to only contain those 2 cell types
                downsample_subgraph = nx.subgraph(downsample_graph, selected_nodes_total)

                # Now with this downsampled subgraph, we perform a fisher's test with a contingency table
                # The first idCL or "idCLs[i]" will be the "target" one
                p_value = run_fisher_exact_test(downsample_subgraph, idCLs[i])

                # If p value is significant, move on to the next comparison
                if p_value < p_threshold:
                    continue
                # If p value is NOT significant, then calculate the proportion of each node in the server subgraph
                else:
                    proportion_cl1 = len(selected_nodes_cl1)/len(selected_nodes_total)
                    proportion_cl2 = len(selected_nodes_cl2)/len(selected_nodes_total)

                    # If the p value was insignificant, then try 3 conditions out. After each condition, update the dataframe and return
                    # The new list of idCLs
                    # For ANY condition that is triggered, RETURN the function immediately containing the modified list of idCLs

                    # Condition 1: (1/epsilon) <= Prop.CL1 / Prop.CL2 <= epsilon
                    if ((1/epsilon) <= (proportion_cl1/proportion_cl2) <= epsilon):
                        # If we cannot distinguish between CL1 and CL2,
                        # Combine the two CL1 & CL2 to be CL1_CL2, and likewise for their celltype names
                        combined_idCL_name = idCLs[i] + "_" + idCLs[j]
                        combined_celltype_name = ab_lookup_dict[idCLs[i]] + "_" + ab_lookup_dict[idCLs[j]]

                        # Update the celltype names for those cells FIRST, since we use the idCLs to refer them
                        # Update the cell labels (idCLs) cells originally listed with CL1 and CL2
                        cell_labels_df.loc[cell_labels_df['labels'] == idCLs[i], 'celltype'] = combined_celltype_name
                        cell_labels_df.loc[cell_labels_df['labels'] == idCLs[j], 'celltype'] = combined_celltype_name
                        
                        print(f"Replacing all cells of {idCLs[i]} with merged name: {combined_idCL_name}")
                        print(f"Replacing all cells of {idCLs[j]} with merged name: {combined_idCL_name}")
                        cell_labels_df.loc[cell_labels_df['labels'] == idCLs[i], 'labels'] = combined_idCL_name
                        cell_labels_df.loc[cell_labels_df['labels'] == idCLs[j], 'labels'] = combined_idCL_name

                        # We must also update all nodes in the downsampled graph that belong to Cl1 or Cl2 with the new
                        # combined celltype name
                        for node in downsample_graph.nodes():
                            # Check if the node has the attribute 'celltype'
                            if 'celltype' in downsample_graph.nodes[node]:
                                # Check if the celltype is 'CL1' or 'CL2'
                                if downsample_graph.nodes[node]['celltype'] == idCLs[i] or downsample_graph.nodes[node]['celltype'] == idCLs[j]:
                                    # Replace the celltype with 'CL1_CL2'
                                    downsample_graph.nodes[node]['celltype'] = combined_idCL_name
                        
                        # Update the OWL graph to contain the merged name for CL1 and CL2
                        mapping_merged_names = {idCLs[i]: combined_idCL_name,
                                                idCLs[j]: combined_idCL_name}
                        nx.relabel_nodes(owl_graph, mapping_merged_names, copy=False) # do it in place
                        
                        # Remove the CL1 and CL2 from the list of idCLs          
                        modified_idCLs.remove(idCLs[i])
                        modified_idCLs.remove(idCLs[j])

                        # Add the new combined idCL
                        modified_idCLs.append(combined_idCL_name)
                        return modified_idCLs, owl_graph
                        
                    elif (proportion_cl1 < proportion_cl2):
                        # Replace all cells from CL1 with labels of CL2

                        # Update the celltype names for CL1 with the ones for CL2
                        idCL1_name = ab_lookup_dict[idCLs[i]]
                        idCL2_name = ab_lookup_dict[idCLs[j]]
                        cell_labels_df.loc[cell_labels_df['labels'] == idCLs[i], 'celltype'] = idCL2_name

                        # Update the cell labels (idCLs) for CL1 with the ones for CL2
                        print(f"Replacing all cells of {idCLs[i]} with {idCLs[j]}")
                        cell_labels_df.loc[cell_labels_df['labels'] == idCLs[i], 'labels'] = idCLs[j]

                        # Also update all nodes in downsampled graph that belong to CL1 to now become CL2
                        for node in downsample_graph.nodes():
                            # Check if the node has the attribute 'celltype'
                            if 'celltype' in downsample_graph.nodes[node]:
                                # Check if the celltype is CL1
                                if downsample_graph.nodes[node]['celltype'] == idCLs[i]:
                                    # Replace the celltype with CL2
                                    downsample_graph.nodes[node]['celltype'] = idCLs[j]

                        # Remove CL1 from the list of idCLs
                        modified_idCLs.remove(idCLs[i])
                        return modified_idCLs, owl_graph

                    elif (proportion_cl2 < proportion_cl1):
                        # Replace all cells from CL2 with labels of CL1

                        # Update the celltype names for CL2 with the ones for CL1
                        idCL1_name = ab_lookup_dict[idCLs[i]]
                        idCL2_name = ab_lookup_dict[idCLs[j]]
                        cell_labels_df.loc[cell_labels_df['labels'] == idCLs[j], 'celltype'] = idCL1_name

                        # Update the cell labels (idCLs) for CL2 with the ones for CL1
                        print(f"Replacing all cells of {idCLs[j]} with {idCLs[i]}")
                        cell_labels_df.loc[cell_labels_df['labels'] == idCLs[j], 'labels'] = idCLs[i]

                        # Also update all nodes in downsampled graph that belong to CL2 to now become CL1
                        for node in downsample_graph.nodes():
                            # Check if the node has the attribute 'celltype'
                            if 'celltype' in downsample_graph.nodes[node]:
                                # Check if the celltype is CL1
                                if downsample_graph.nodes[node]['celltype'] == idCLs[j]:
                                    # Replace the celltype with CL2
                                    downsample_graph.nodes[node]['celltype'] = idCLs[i]

                        # Remove CL2 from the list of idCLs
                        modified_idCLs.remove(idCLs[j])
                        return modified_idCLs, owl_graph
                    
            # If the distance between two nodes is greater than 2, ignore and move onto next comparison
            else:
                continue

    return modified_idCLs, owl_graph

def keep_calling_part2(owl_graph, downsample_graph, idCLs, cell_labels_df, p_threshold, epsilon):
    """
    This calls our part2_filter function over and over

    Parameters:
        owl_graph from the class
        downsample_graph from the class
        idCLs from the object
        cell labels from the object
        
    """
    # Initialize modified_idCLs with an empty list
    modified_idCLs = []
    
    # Initialize a flag to track changes
    changed = True
    
    # Keep looping until no changes are made
    while changed:
        # Call part2_filter function and get the modified_idCLs list
        modified_idCLs, modified_owl = part2_filter(owl_graph, 
                                                    downsample_graph, 
                                                    idCLs, cell_labels_df, 
                                                    p_threshold=p_threshold, 
                                                    epsilon=epsilon)
        
        # Check if the current modified_idCLs is the same as the previous one
        if modified_idCLs == idCLs:
            # If they are the same, set the flag to False to exit the loop
            changed = False
        else:
            # If they are different, update idCLs with the new modified_idCLs
            idCLs = modified_idCLs
            # Also use the updated OWL graph
            owl_graph = modified_owl
    
    # Return the final modified_idCLs
    return modified_idCLs

#   Part 3 Filtering (Requires run_stvea() output: nearest neighbor distances)
def calculate_D1(nn_dist):
    results = pd.DataFrame()
    # Find all nearest neighbors (non-zero value) for each row (query cell)
    # Take the sum of all the distances to each nearest neighbor
    non_zero_sums = nn_dist[nn_dist != 0].sum(axis=1)

    # Count the number of neighbors for each row
    non_zero_counts = (nn_dist != 0).sum(axis=1)

    # Find the average for each row
    average_non_zero_values = non_zero_sums / non_zero_counts
    results['avg_nn_distance'] = average_non_zero_values

    return results

def calculate_D2_fast(nn_dist):
    # List to hold all average pairwise distances for each row/query cell
    avg_pairwise_distances = []

    # Number of query cells to iterate over
    num_rows = len(nn_dist.index)

    for row_index in range(num_rows):
        row = nn_dist.iloc[row_index]
        # Isolate non-zero values for the selected row. These are the nearest neighbors for that cell/row
        non_zero_values_row = row[row != 0]

        # Convert non_zero_values_row to a numpy array
        neighbor_distances = non_zero_values_row.values

        # Compute the number of neighbors
        num_neighbors = len(neighbor_distances)

        if num_neighbors > 1:
            # Compute pairwise distances using broadcasting
            pairwise_distances = np.abs(neighbor_distances[:, np.newaxis] - neighbor_distances[np.newaxis, :])

            # Compute the sum of all pairwise values in the upper triangle (excluding diagonal)
            sum_pairwise_values = np.sum(np.triu(pairwise_distances, k=1))

            # Count the number of pairwise values in the upper triangle
            num_pairwise_values = (num_neighbors * (num_neighbors - 1)) / 2

            # Compute the average of all pairwise values
            average_pairwise_value = sum_pairwise_values / num_pairwise_values
        else:
            average_pairwise_value = 0

        avg_pairwise_distances.append(average_pairwise_value)

    d2_df = pd.DataFrame(index=nn_dist.index, data=avg_pairwise_distances, columns=["avg_pairwise_distance"])
    return d2_df

def calculate_D1_D2_ratio(D1, D2):
    # Combine the two dataframes
    combined = pd.concat([D1, D2], axis=1)

    # Calculate the ratio now for nn_distance/pairwise_distance
    combined['ratio'] = combined['avg_nn_distance'] / combined['avg_pairwise_distance']
    return combined

#   Part 4 Filtering (Requires run_stvea() output: transfer matrix and imputed reference dataset from server)
def group_cells_by_type(imputed_reference):
    cell_indices_by_type = {}
    
    # Single out the 'idCL' column
    protein_idCLs = imputed_reference['idCL']

    # Group all rows by cell type
    grouped = protein_idCLs.to_frame().groupby('idCL')
    
    # Add to dictionary
    for cell_type, group in grouped:
        cell_indices_by_type[cell_type] = group.index

    return cell_indices_by_type

def calculate_entropies_fast(transfer_matrix, cell_indices_by_type):
    # Create a DataFrame to hold the summed probabilities for each cell type
    cell_type_sums = pd.DataFrame(index=transfer_matrix.index)

    # Compute the sums of the probabilities for each cell type
    for cell_type, indices in cell_indices_by_type.items():
        # print(indices)
        cell_type_sums[cell_type] = transfer_matrix[indices].sum(axis=1)
    
    # Calculate entropy for each query cell
    entropies = cell_type_sums.apply(lambda row: entropy(row), axis=1)
    
    # Convert the result to a DataFrame
    entropies_df = entropies.to_frame(name='entropy')
    
    return entropies_df

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
        self._downsample_pairwise_graph = None
        self._nn_dist = None

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
            try:
                owl_link = _update_cl_owl()
                G_nxo = from_file(owl_link)
            except:
                owl_link = str(files('immunopheno.data').joinpath('cl_2024_05_15.owl'))
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
            raise Exception(exp_response.text)
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
            raise Exception(wa_response.text)
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
            raise Exception(wc_response.text)
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
            raise Exception(we_response.text)
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
        
        IPD_new = copy.deepcopy(IPD)

        # Check if reference query parameters have changed OR if the reference table is empty
        if (IPD_new, IPD_new._stvea_correction_value, idBTO, idExperiment, 
            parse_option, rho, population_size) != self._last_stvea_params or self.imputed_reference is None:

            antibody_pairs = [[key, value] for key, value in IPD_new._ab_ids_dict.items()]
        
            stvea_body = {
                "antibody_pairs": antibody_pairs,
                "idBTO": idBTO,
                "idExperiment": idExperiment,
                "parse_option": parse_option,
                "population_size": population_size
            }

            print("Retrieving reference dataset...")
            timeout = (600, 600) # connection, read timeout
            stvea_response = requests.post(f"{self.url}/api/stveareference", json=stvea_body, timeout=timeout)
            if 'text/html' in stvea_response.headers.get('content-type'):
                raise Exception(stvea_response.text)
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
                        lambda x: x - IPD_new._stvea_correction_value if (x != 0 and type(x) is not str) else x)

            # Store these parameters to check for subsequent function calls
            self._last_stvea_params = (IPD_new, IPD_new._stvea_correction_value, idBTO, idExperiment, 
                                       parse_option, rho, population_size)

        # Separate out the antibody counts from the cell IDs 
        imputed_antibodies = self.imputed_reference.loc[:, self.imputed_reference.columns != 'idCL']
        imputed_idCLs = self.imputed_reference['idCL'].to_frame()
    
        # Convert antibody names from CODEX normalized counts to their IDs
        codex_normalized_with_ids = IPD_new._normalized_counts_df.rename(columns=IPD_new._ab_ids_dict, inplace=False)
    
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
        
        # Store original nearest neighbor distances in class
        nn_dist = cn.stvea.nn_dist_matrix
        self._nn_dist = nn_dist

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

        # labels_df now contains two columns: 'labels' and 'celltype'
        # However, some rows may have been filtered out during the run_cca step
        # Fill in missing rows from the original dataset with "Not Assigned"
        index_diff = codex_normalized_with_ids.index.difference(labels_df.index)
        filtered_cells = {'labels': ['Not Assigned'] * len(index_diff), 'celltype': ['Not Assigned'] * len(index_diff)}
        new_df = pd.DataFrame(filtered_cells, index=index_diff)
        complete_labels_df = labels_df.append(new_df)

        # Ensure that all NaN values are replaced with "Not Assigned" in the "celltype" column
        complete_labels_df['celltype'] = complete_labels_df['celltype'].fillna('Not Assigned')
        
        # Before setting norm_cell_types, check if it matches the previous. If not, reset norm_umap field
        if not (complete_labels_df.equals(IPD_new._cell_labels_filt_df)):
            IPD_new._norm_umap = None
        IPD_new._cell_labels_filt_df = complete_labels_df
        
        # Add labels to raw cell labels as well. The filtered rows will be marked as "filtered_by_stvea"
        original_cells_index = IPD_new.protein.index
        merged_df = IPD_new._cell_labels_filt_df.reindex(original_cells_index)
        merged_df = merged_df.fillna("filtered_by_stvea")

        # Check if the raw and norm labels have changed. If so, reset the UMAP field in IPD
        if not (merged_df.equals(IPD_new._cell_labels)):
            IPD_new._raw_umap = None
        IPD_new._cell_labels = merged_df

        # Make sure the indexes match
        IPD_new._normalized_counts_df = IPD_new._normalized_counts_df.loc[IPD_new._cell_labels_filt_df.index]
        print("Annotation transfer complete.")
        return IPD_new

    def _part1_localization(self, IPD, p_threshold=0.05):
        """
        Remove: the option to either remove cells completely from the object after filtering
                OR set it to "Not Assigned" instead in the cell annotations
        """
        # We will be working with the normalized_counts and labels in the IPD object
        # First, we will need to ignore all initial cells labeled as "Not Assigned"
        filtered_index = IPD.labels[IPD.labels['labels'] != 'Not Assigned'].index
        norm_counts = pd.DataFrame(IPD.normalized_counts.loc[filtered_index])
        norm_labels = pd.DataFrame(IPD.labels.loc[filtered_index])
    
        # Temporarily rename the "labels" column to "idCL"
        norm_counts_labels = norm_labels.rename(columns={"labels":"idCL"})
        
        # Combine the normalized protein with the labels field containing "idCL"
        norm_combine = pd.concat([norm_counts, norm_counts_labels.loc[norm_labels.index]["idCL"]], axis = 1)
    
        ## GRAPH CREATION ## 
        # Create dictionary of cell barcodes to their cell types
        if self._downsample_pairwise_graph is None:
            ct_lookup = norm_counts_labels["idCL"].to_dict()
        
            # Downsample to 5000 cells and get pairwise distance matrix
            print("Downsampling dataset to 5000 cells...")
            ds_norm = downsample(norm_combine, downsample_size=5000)
        
            # Calculate the pairwise correlation distances between all rows (cells)
            print("Calculating pairwise distances between cells...")
            df_without_idCL = ds_norm.drop(columns=["idCL"])
            pairwise_correlation = (1 - df_without_idCL.T.corr())
        
            # Create adjacency matrix
            adj_mat = pearson_correlation_adjaceny_matrix(pairwise_correlation)
        
            # Create graph
            print("Generating cell graph...")
            G = fast_cell_label_graph(adj_mat)
        
            # Set cell type for each node in graph
            nx.set_node_attributes(G, ct_lookup, "celltype")
        
            ###################################### Store this graph object in the class
            self._downsample_pairwise_graph = G
        else:
            G = self._downsample_pairwise_graph
    
        # Get all unique cell types to filter over
        all_celltypes = set(norm_counts_labels["idCL"])
    
        # Store labels to remove
        labels_to_remove = []

        print("Conducting fisher exact test for each cell type...")
        # Perform fisher for every cell type
        for celltype in all_celltypes:
            p_val = run_fisher_exact_test(G, celltype)
            if p_val > p_threshold:
                labels_to_remove.append(celltype)
    
        ## RENAME step
        print(f"Number of cell types to rename:", len(labels_to_remove))
        
        affected_cells = norm_counts_labels[norm_counts_labels['idCL'].isin(labels_to_remove)]
        affected_cells_index = list(affected_cells.index)

        # Find which cells contain these labels
        grouped = affected_cells['idCL'].to_frame().groupby('idCL')
        for cell_type, group in grouped:
            if cell_type == "Not Assigned":
                continue
            print(f"{cell_type}: {len(group.index)}")
    
        # Make a copy of the object to return
        temp_copy = copy.deepcopy(IPD)
    
        # This function needs to reset the UMAPs present. So remove _raw_umap and _norm_umap and set to None
        temp_copy._raw_umap = None
        temp_copy._norm_umap = None
        temp_copy._umap_kwargs = None
        
        print(f"Renaming {len(affected_cells_index)} cells with 'Not Assigned' label...")
        temp_copy.labels.loc[affected_cells.index, ['labels', 'celltype']] = "Not Assigned" # Modify the normalized cell labels
        temp_copy._cell_labels.loc[affected_cells.index, ['labels', 'celltype']] = "Not Assigned" # Modify the unnormalized cell labels
        print(f"Renamed {len(affected_cells_index)} cells.")
        return temp_copy

    def _part2_merging(self, IPD, p_threshold=0.05, epsilon=4):
        # Create a deepcopy of the OWL graph. This one will be constantly updated
        owl_graph_deepcopy = copy.deepcopy(self._OWL_graph)

        # If the downsampled pairwise distance graph doesn't exist, create it now
        if self._downsample_pairwise_graph is None:
            # We will be working with the normalized_counts and labels in the IPD object
            # First, we will need to ignore all initial cells labeled as "Not Assigned"
            filtered_index = IPD.labels[IPD.labels['labels'] != 'Not Assigned'].index
            norm_counts = pd.DataFrame(IPD.normalized_counts.loc[filtered_index])
            norm_labels = pd.DataFrame(IPD.labels.loc[filtered_index])
        
            # Temporarily rename the "labels" column to "idCL"
            norm_counts_labels = norm_labels.rename(columns={"labels":"idCL"})
            
            # Combine the normalized protein with the labels field containing "idCL"
            norm_combine = pd.concat([norm_counts, norm_counts_labels.loc[norm_labels.index]["idCL"]], axis = 1)
        
            ## GRAPH CREATION ## 
            # Create dictionary of cell barcodes to their cell types
            ct_lookup = norm_counts_labels["idCL"].to_dict()
        
            # Downsample to 5000 cells and get pairwise distance matrix
            print("Downsampling dataset to 5000 cells...")
            ds_norm = downsample(norm_combine, downsample_size=5000)
        
            # Calculate the pairwise correlation distances between all rows (cells)
            print("Calculating pairwise distances between cells...")
            df_without_idCL = ds_norm.drop(columns=["idCL"])
            pairwise_correlation = (1 - df_without_idCL.T.corr())
        
            # Create adjacency matrix
            adj_mat = pearson_correlation_adjaceny_matrix(pairwise_correlation)
        
            # Create graph
            print("Generating cell graph...")
            G = fast_cell_label_graph(adj_mat)
        
            # Set cell type for each node in graph
            nx.set_node_attributes(G, ct_lookup, "celltype")
        
            ###################################### Store this graph object in the class
            self._downsample_pairwise_graph = G
        
        # Create a deepcopy of the downsampled pairwise distance graph. It will be updated as well
        downsample_graph_deepcopy = copy.deepcopy(self._downsample_pairwise_graph)
    
        # Remove all nodes in OWL graph that are not "CL:" cell ontology
        nodes_to_remove = [node for node in owl_graph_deepcopy.nodes() if "CL:" not in node]
        owl_graph_deepcopy.remove_nodes_from(nodes_to_remove)
    
        # Get all cell types from the IPD object, ignoring "Not Assigned"
        idCLs = list(set(IPD.labels['labels']))
        if "Not Assigned" in idCLs:
            idCLs.remove("Not Assigned")
        cell_labels_df = IPD.labels.copy(deep=True)
    
        # Call filter function until no more labels are filtered out. This modifies cell_labels_df in-place
        celltypes_remaining = keep_calling_part2(owl_graph_deepcopy.to_undirected(), 
                                                 downsample_graph_deepcopy, 
                                                 idCLs, 
                                                 cell_labels_df, 
                                                 p_threshold, 
                                                 epsilon)
        
        number_merged = sum(1 for element in celltypes_remaining if "_" in element)
        print(f"Number of cell types merged: {number_merged}")
    
        # Create a new object to return
        temp_copy = copy.deepcopy(IPD)
    
        # This function needs to reset the UMAPs present. So remove _raw_umap and _norm_umap and set to None
        temp_copy._raw_umap = None
        temp_copy._norm_umap = None
        temp_copy._umap_kwargs = None
    
        # Set the new cell_labels_df in the temp object to return
        temp_copy.labels = cell_labels_df
    
        return temp_copy
    
    def _part3_reassign_by_distance_ratio(self, IPD, distance_ratio_threshold=10):
        if self._nn_dist is None:
            raise Exception("Missing distance matrix. run_stvea() must be called to filter by distance ratios.")

        cell_labels_df = IPD.labels
        
        # D1 values
        d1_df = calculate_D1(self._nn_dist)

        # D2 values
        d2_df = calculate_D2_fast(self._nn_dist)

        # D1/D2 ratio
        ratio_df = calculate_D1_D2_ratio(d1_df, d2_df)        

        # Find all rows/cells with a ratio greater than the threshold
        cells_to_unlabel = ratio_df[ratio_df["ratio"] > distance_ratio_threshold]

        # Re-assign all affected cells with "Not Assigned"
        temp_labels = cell_labels_df.copy(deep=True)
        
        # Find the number of cells for each cell type getting re-assigned
        cells_group_df = (temp_labels.loc[cells_to_unlabel.index])
        # Disregard cells that were already named "Not assigned"
        num_cells_already_not_assigned = sum(cells_group_df["labels"] == "Not Assigned")
        print("Number of cells renamed to 'Not Assigned':", len(cells_group_df) - num_cells_already_not_assigned )
        
        # Group all rows by cell type
        grouped = cells_group_df.groupby('labels')
        for cell_type, group in grouped:
            if cell_type == "Not Assigned":
                continue
            print(f"{cell_type}: {len(group.index)}")

        # Renaming step
        temp_labels.loc[cells_to_unlabel.index, ['labels', 'celltype']] = "Not Assigned"

        # Set this temp_labels table in a new IPD object
        temp_copy = copy.deepcopy(IPD)
        # This function needs to reset the UMAPs present. So remove _raw_umap and _norm_umap and set to None
        temp_copy._raw_umap = None
        temp_copy._norm_umap = None
        temp_copy._umap_kwargs = None

        # Set the new cell_labels_df in the temp object to return
        temp_copy.labels = temp_labels
        
        return temp_copy
    
    def _part4_reassign_by_entropy(self, IPD, entropy_threshold=10):
        if self.transfer_matrix is None or self.imputed_reference is None:
            raise Exception("Missing transfer_matrix/imputed_reference. run_stvea() must be called to filter by entropies.")
        
        cell_labels_df = IPD.labels
        
        # Find cell indices for each cell type
        cell_indices_by_type = group_cells_by_type(self.imputed_reference)

        # Calculate entropies for each cell type for each query cell in transfer matrix
        entropies_df = calculate_entropies_fast(self.transfer_matrix, cell_indices_by_type)
        
        # Find all rows/cells with a ratio greater than the threshold
        cells_to_unlabel = entropies_df[entropies_df["entropy"] > entropy_threshold]
        
        # Re-assign all affected cells with "Not Assigned"
        temp_labels = cell_labels_df.copy(deep=True)
        
        # Find the number of cells for each cell type getting re-assigned
        cells_group_df = (temp_labels.loc[cells_to_unlabel.index])
        # Disregard cells that were already named "Not assigned"
        num_cells_already_not_assigned = sum(cells_group_df["labels"] == "Not Assigned")
        print("Number of cells renamed to 'Not Assigned':", len(cells_group_df) - num_cells_already_not_assigned )
        
        # Group all rows by cell type
        grouped = cells_group_df.groupby('labels')
        for cell_type, group in grouped:
            if cell_type == "Not Assigned":
                continue
            print(f"{cell_type}: {len(group.index)}")

        # Renaming step
        temp_labels.loc[cells_to_unlabel.index, ['labels', 'celltype']] = "Not Assigned"

        # Set this temp_labels table in a new IPD object
        temp_copy = copy.deepcopy(IPD)
        # This function needs to reset the UMAPs present. So remove _raw_umap and _norm_umap and set to None
        temp_copy._raw_umap = None
        temp_copy._norm_umap = None
        temp_copy._umap_kwargs = None

        # Set the new cell_labels_df in the temp object to return
        temp_copy.labels = temp_labels
        
        return temp_copy
    
    def plot_filter_metrics(self):
        # Check if transfer_matrix and imputed_reference exists
        if self._nn_dist is None or self.transfer_matrix is None or self.imputed_reference is None:
            raise Exception("Missing transfer_matrix. run_stvea() must be called to use plot_filter_metrics()")
        
        # Plot distance D1/D2 ratios
        # D1 values
        d1_df = calculate_D1(self._nn_dist)

        # D2 values
        d2_df = calculate_D2_fast(self._nn_dist)

        # D1/D2 ratio
        ratio = calculate_D1_D2_ratio(d1_df, d2_df)        

        # Plot a histogram of all ratio values to decide on the ratio_threshold
        ratio_fig = px.histogram(ratio["ratio"], title="D1/D2 Ratios For All Query Cells" + 
                            "<br>" +
                            "<sup>D1: Average distance between nearest neighbor and query cell</sup>" +
                            "<br>" + 
                            "<sup>D2: Average pairwise distance among nearest neighbors</sup>").update_layout(height=500)
        ratio_fig.show()

        # Plot entropy values
        # Find cell indices for each cell type
        cell_indices_by_type = group_cells_by_type(self.imputed_reference)

        # Calculate entropies for each cell type for each query cell in transfer matrix
        entropies_df = calculate_entropies_fast(self.transfer_matrix, cell_indices_by_type)

        entropy_fig = px.histogram(entropies_df["entropy"], title="Entropies For All Query Cells").update_layout(height=500)
        entropy_fig.show()

    def filter_labels(self, 
                      IPD,
                      localization=False,               # Filtering step
                      merging=False,                    # Filtering step
                      distance_ratio=False,             # Filtering step
                      entropy=False,                    # Filtering step
                      p_threshold_localization=0.05,    # P value threshold for fisher exact test during localization 
                      p_threshold_merging=0.05,         # P value threshold during merging
                      epsilon_merging=4,                # Epsilon value for deciding to merge two cell types based on proportion ratio
                      distance_ratio_threshold=2,       # Ratio threshold when filtering cells by NN distance ratios (D1/D2)
                      entropy_threshold=2,              # Entropy threshold when filtering cells by total entropy for cell types
                      remove_labels=False):             # Remove cells as "Not Assigned" at the end of filtering:             
        
        # Ensure at least one of the filtering steps is enabled
        if not (localization or merging or distance_ratio or entropy):
            raise ValueError("At least one of 'localization', 'merging', 'distance_ratio', or 'entropy' must be set to True.")
        
        # Issue warnings for unused parameters
        if not localization:
            logging.warning("Parameters for localization will be ignored because 'localization' is set to False.")
        
        if not merging:
            logging.warning("Parameters for merging will be ignored because 'merging' is set to False.")
        
        if not distance_ratio:
            logging.warning("Parameters for distance_ratio will be ignored because 'distance_ratio' is set to False.")
        
        if not entropy:
            logging.warning("Parameters for entropy will be ignored because 'entropy' is set to False.")
        
        # Start with the initial IPD
        IPD_new = copy.deepcopy(IPD)
        
        # Call the localization function if needed
        if localization:
            print("Performing localization...")
            IPD_new = self._part1_localization(IPD=IPD_new, p_threshold=p_threshold_localization)
            print("Localization complete.\n")
        # Call the merging function if needed
        if merging:
            print("Performing merging...")
            IPD_new = self._part2_merging(IPD=IPD_new, p_threshold=p_threshold_merging, epsilon=epsilon_merging)
            print("Merging complete.\n")
        # Call the distance_ratio function if needed
        if distance_ratio:
            print("Performing nearest neighbor distance ratio filtering...")
            IPD_new = self._part3_reassign_by_distance_ratio(IPD=IPD_new, distance_ratio_threshold=distance_ratio_threshold)
            print("Distance_ratio filtering complete.\n")
        # Call the entropy function if needed
        if entropy:
            print("Performing cell type entropy filtering...")
            IPD_new = self._part4_reassign_by_entropy(IPD=IPD_new, entropy_threshold=entropy_threshold)
            print("Entropy filtering complete.\n")
        
        # Make sure all "Not Assigned" rows are consistent in both "labels" and "celltypes" column of IPD.labels
        IPD_new.labels.loc[IPD_new.labels["labels"] == "Not Assigned", "celltype"] = "Not Assigned"
        affected_cells_index = IPD_new.labels[IPD_new.labels["labels"] == "Not Assigned"].index
        
        # Remove all rows/cells that had "Not Assigned"
        if remove_labels:
            print(f"Removing {len(affected_cells_index)} cells with 'Not Assigned' from object...")
            # Filter out attributes that start with an underscore and are not properties
            attrs_to_iterate = [attr_name for attr_name in dir(IPD_new) if attr_name.startswith('_') and not isinstance(getattr(IPD_new.__class__, attr_name, None), property)]
            
            for attr_name in attrs_to_iterate:
                attr_value = getattr(IPD_new, attr_name)
                if isinstance(attr_value, pd.DataFrame):
                    # If the attribute is a DataFrame, drop rows from it using the provided index
                    setattr(IPD_new, attr_name, attr_value.drop(index=affected_cells_index, errors='ignore'))
            
            print(f"Removed {len(affected_cells_index)} cells from object.")
        
        return IPD_new
    
    def db_stats(self):
        stats_res = requests.get(f"{self.url}/api/databasestatistics")  
        if 'text/html' in stats_res.headers.get('content-type'):
            raise Exception(stats_res.text)
        elif 'application/json' in stats_res.headers.get('content-type'):
            stats_JSON = stats_res.json()

        print("Database Statistics\n===================")
        print("Number of experiments:", stats_JSON["num_exp"])
        print("Number of tissues:", stats_JSON["num_tissue"])
        print("Number of cells:", stats_JSON["num_cells"])
        print("Number of antibodies:", stats_JSON["num_ab"])
        print("Number of antibody targets:", stats_JSON["num_targets"])
        print("Number of antibody clones:", stats_JSON["num_clones"])
        print("Average number of experiments per antibody:", "{:.2f}".format(stats_JSON["avg_exp"]))
