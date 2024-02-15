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

class ImmunoPhenoDB_Connect:
    def __init__(self, url: str):
        self.url = url
        self._OWL_graph = None
        self._subgraph = None
        self._db_idCLs = None
        self._db_idCL_names = None

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

                idCL_names = {}
                # Find all readable cell type names for each node
                for node in list(self._subgraph.nodes):
                    idCL_names[node] = convert_idCL_readable(node)
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