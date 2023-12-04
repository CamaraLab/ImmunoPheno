import numpy as np
import requests
import urllib.parse
import random
import json
import logging
import warnings
import pandas as pd

import plotly.express as px

import networkx as nx
from networkx.exception import NetworkXError
from nxontology.imports import from_file

import matplotlib.pyplot as plt
from netgraph import Graph

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def _update_cl_owl():
    warnings.filterwarnings("ignore")
    response = requests.get('https://www.ebi.ac.uk/ols4/api/ontologies/cl')
    owl_link = response.json()['config']['versionIri']
    return owl_link

def subgraph(G, nodes, plot=False):
    root='CL:0000000'
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

    if plot:
        if nx.is_tree(subgraph) and len(all_LCA) == 1:
            pos = hierarchy_pos(subgraph, all_LCA[0])  
             
            # Define color map
            color_map = ["#FCC8C8" if node in nodes else "#C8ECFC" for node in subgraph] 
            plt.figure(figsize=(4, 4))
            nx.draw(subgraph,
                    pos=pos, 
                    font_size=7, 
                    node_size=[len(v)**2 * 20 for v in subgraph.nodes()],    
                    node_color=color_map, 
                    with_labels=True)
            plt.show()
            
        else:
            node_color = dict()
            for node in subgraph.nodes:
                node_color[node] = "#FCC8C8" if node in nodes else "#C8ECFC"
            
            plt.figure(figsize=(20, 20))
            
            Graph(subgraph, 
                  node_layout = 'dot', 
                  node_color = node_color, 
                  node_size=(3), 
                  node_labels=True, 
                  node_label_fontdict=dict(size=9),
                  edge_color='black', 
                  edge_layout='straight', 
                  edge_label_fontdict=dict(fontweight='bold'), 
                  node_edge_width=0.1, 
                  edge_width=0.3, 
                  arrows=True)
            plt.show()
        
    return subgraph

def convert_idCL_readable(idCL:str):
    idCL_params = {
        'q': idCL,
        'exact': 'true',
        'ontology': 'cl',
        'fieldList': 'label',
        'rows': 1,
        'start': 0
    }
    
    res = requests.get("https://www.ebi.ac.uk/ols4/api/search", params=idCL_params)
    res_JSON = res.json()
    cellType = res_JSON['response']['docs'][0]['label']
    
    return cellType

def convert_ab_readable(ab_id:str):
    res = requests.get("http://www.scicrunch.org" + "/resolver/" + ab_id + ".json")
    res_JSON = res.json()
    ab_name = (res_JSON['hits']['hits'][0]
                    ['_source']['antibodies']['primary'][0]
                    ['targets'][0]['name'])
    return ab_name

class ImmunoPhenoDB_Connect:
    def __init__(self, url: str):
        self.url = url
        self._OWL_graph = None
        self._subgraph = None
        self._db_idCLs = None
        self._descendants = None
        self._lmm_results = None

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
    
    def plot_db_graph(self):
        subgraph(self._OWL_graph, self._db_idCLs, plot=True)

    def find_antibodies(self, 
                        id_CLs: list, 
                        idBTO: list = None, 
                        idExperiment: list = None, 
                        plot: bool = False)-> pd.DataFrame: 
        
        # First find all descendants of the provided id_CLs. These will be included 
        # when running the LMM
        try: 
            node_fam_dict = self._find_descendants(id_CLs)
        except NetworkXError as err:
            err_msg = str(err).split(' ')[2] # Get idCL error
            raise Exception(f"Error. {err_msg} not found in the database")
        
        # Call API endpoint here to get_Abs to return dataframe        
        abs_body = {
            "idCL": node_fam_dict,
            "idBTO": idBTO,
            "idExperiment": idExperiment
        }
        
        abs_response = requests.post(f"{self.url}/api/findabs", json=abs_body)

        # Check response from server
        if 'text/html' in abs_response.headers.get('content-type'):
            res_df = abs_response.text
            return res_df
        elif 'application/json' in abs_response.headers.get('content-type'):
            res_JSON = abs_response.json()
            res_df = pd.DataFrame.from_dict(res_JSON)
        
        # If plotting, call API endpoint to recieve dictionary of dataframes
        # We will need the antibodies from the result df and node_fam_dict
        if isinstance(res_df, pd.DataFrame) and plot is True:
            idCL_plots = {}
            
            antibodies = list(res_df.index) # antibodies to send to endpoint

            for idCL_parent, descendants in node_fam_dict.items():
                desc_copy = descendants.copy()
                desc_copy.insert(0, idCL_parent) # all idCLs to send to endpoint

                plot_abs_body = {
                    "abs": antibodies,
                    "idcls": desc_copy
                }
                
                abs_plot_response = requests.post(f"{self.url}/api/plotabs", json=plot_abs_body) # send antibodies and idCLs
                abs_plot_JSON = abs_plot_response.json() # returns a dictionary
                abs_plot_df = pd.DataFrame.from_dict(abs_plot_JSON) # Convert into a dataframe
                
                # Add to dictionary
                idCL_plots[idCL_parent] = abs_plot_df
            
            # Change x axis to use common antibody names
            ab_targets = list(res_df['target'])
            mapping_dict = dict(zip(res_df.index, ab_targets))

            modified_mapping_dict = {}
                
            # Add antibody IDs next to their targets in the X-axis
            for i in mapping_dict:
                modified_mapping_dict[i] = str(i) + f" ({mapping_dict[i]})"
            new_ab_targets = list(modified_mapping_dict.values())
            
            # Plot using plotly
            for idCL, df in idCL_plots.items():
                # Convert idCL to common name
                idCL_readable = convert_idCL_readable(idCL)
                
                # Rename antibodies to common name
                df['ab_target'] = df['idAntibody'].map(modified_mapping_dict)
                                
                # Rename normValue to z score
                df.rename(columns={'normValue': 'z score'}, inplace=True)
                
                # Plot
                fig = px.violin(data_frame = df, 
                                x = 'ab_target', 
                                y = 'z score', 
                                color = 'ab_target', 
                                category_orders={'ab_target' : new_ab_targets},
                                box=True)
                fig.update_layout(title_text=f"Antibodies for: {idCL_readable}", 
                                  title_x = 0.45, 
                                  font=dict(size=8), 
                                  width=1400, 
                                  height=600)
                fig.update_traces(width=0.75, selector=dict(type='violin'))
                fig.update_traces(marker={'size': 1})
                fig.show()
    
        return res_df
    
    def find_celltypes(self,
                       ab_ids: list,
                       idBTO: list = None, 
                       idExperiment: list = None,
                       plot: bool = False) -> dict: 
        ab_df = {}
        
        celltypes_body = {
            "ab": ab_ids,
            "idBTO": idBTO,
            "idExp": idExperiment
        }
        
        # Call API endpoint here to get dataframe
        celltypes_response = requests.post(f"{self.url}/api/findcelltypes", json=celltypes_body)
        
        if 'text/html' in celltypes_response.headers.get('content-type'):
            res_dict_strjson = celltypes_response.text
            return res_dict_strjson
        elif 'application/json' in celltypes_response.headers.get('content-type'):
            res_dict_strjson = celltypes_response.json() # returns a dict of string-jsons
        
        # Convert all of these string-jsons into dataframes
        for key, value in res_dict_strjson.items():
            # Convert string-json into dict
            temp_dict = json.loads(value)
            
            # Convert dict into dataframe
            temp_df = pd.DataFrame.from_dict(temp_dict)
            
            # Add to final dict of dfs
            ab_df[key] = temp_df
            
        # Check for skipped antibodies If so, then the server found no cells in the database
        res_abs = list(res_dict_strjson.keys())
        missing_abs = list(set(ab_ids) - set(res_abs))
        for missing in missing_abs:
            logging.warning(f"No cells found in the database for {missing}. Skipping {missing}")
    
        # If plotting, call API endpoint to receive dictionary of dataframes
        # Plot here using plotly
        if isinstance(res_dict_strjson, dict) and plot is True:
            plotting_dfs = {} # key: ab, value: df
            
            for key, value in ab_df.items():
                # Extract index from each dataframe
                df_idCLs = list(value.index)
                
                # Send antibody and idCLs to get dataframe
                plot_celltypes_body = {
                    "ab": key,
                    "idcls": df_idCLs
                }
                
                celltypes_plot_response = requests.post(f"{self.url}/api/plotcelltypes", json=plot_celltypes_body)
                celltypes_plot_JSON = celltypes_plot_response.json() # returns a dict
                celltypes_plot_df = pd.DataFrame.from_dict(celltypes_plot_JSON) # Convert into a dataframe
                
                # Store all of these in a dict
                plotting_dfs[key] = celltypes_plot_df
            
            for key, value in ab_df.items():
                # Create a temporary mapping dictionary to change idCL names to common names
                celltype_names = list(value['cellType'])
                mapping_dict = dict(zip(value.index, celltype_names))

                modified_mapping_dict = {}
                
                # Add cell type ID next to readable name
                for i in mapping_dict:
                    modified_mapping_dict[i] = str(i) + f" ({mapping_dict[i]})"
                new_celltype_names = list(modified_mapping_dict.values())
                
                # Convert antibody to common name
                ab_readable = convert_ab_readable(key)
                
                # Get plot df
                plot_df = plotting_dfs[key]
                
                # Rename idCLs to common name
                plot_df['cell_type'] = plot_df['idCL'].map(modified_mapping_dict)

                # Rename normValue to z score
                plot_df.rename(columns={'normValue': 'z score'}, inplace=True)

                # Plot
                fig = px.violin(data_frame=plot_df, 
                                x = 'cell_type', 
                                y = 'z score', 
                                color = 'cell_type', 
                                category_orders={'cell_type' : new_celltype_names}, 
                                box=True)
                fig.update_layout(title_text=f"Cell Types for: {ab_readable} antibody", 
                                  title_x = 0.45, 
                                  font=dict(size=8), 
                                  width=1400, 
                                  height=600)
                fig.update_traces(width=0.75, selector=dict(type='violin'))
                fig.update_traces(marker={'size': 1})
                fig.show()
                
        return ab_df
    
    def find_experiments(self,
                         ab_id: list = None,
                         idCL: list = None,
                         idBTO: list = None) -> pd.DataFrame:
        exp_body = {
            "ab": ab_id,
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