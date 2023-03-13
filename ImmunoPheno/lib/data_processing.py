from models import _nb_mle_results, _gmm_results

import numpy as np

import scipy.stats as ss
import scipy.optimize as so
import scipy.special as sp

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import pandas as pd

import statistics

def clean_adt(protein_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes ADT_ tags (if present) from proteins and transposes the DataFrame

    Parameters:
      protein_df (Pandas DataFrame): protein data, rows = proteins, cols = cells

    Returns:
      protein_df_copy (Pandas DataFrame): transposed df
    """
    
    protein_df_copy = protein_df.copy(deep=True)
    protein_df_copy = protein_df_copy.T

    return protein_df_copy

def clean_rna(gene_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transposes the RNA DataFrame of UMIs containing features (genes) and cells

    Parameters:
        gene_df (Pandas DataFrame): RNA data, rows = features, cols = cells

    Returns:
        gene_df_copy_transposed (Pandas DataFrame): transposed RNA DataFrame
    """

    gene_df_copy = gene_df.copy(deep=True)
    gene_df_copy_transposed = gene_df_copy.T

    return gene_df_copy_transposed

def _log_transform(d_vect: list,
                  scale: (int) = 1) -> list:
    """
    Applies log transformation to a list containing raw values

    Parameters:
        d_vect (list): raw counts from protein data
        scale (int): transformation scale value (optional)
    
    Returns:
        data_array: log-transformed counts
    """
    data_array = np.array([scale*i + 1 - (scale*min(d_vect)) for i in d_vect])
    return np.log(data_array)

def _arcsinh_transform(d_vect: list,
                      scale: int = 1) -> list:
    """
    Applies arcsinh transformation to a list containing raw values

    Parameters:
        d_vect (list): raw counts from protein data
        scale (int): transformation scale value (optional)
    
    Returns:
        data_array: arcsinh-transformed counts
    """
    data_array = np.array([scale*i - (scale*min(d_vect)) for i in d_vect])
    return np.arcsinh(data_array.astype(float))

def _conv_np_mean_stdev(n: float,
                        p: float) -> tuple:
    """
    Converts Negative Binomial n, p parameters into mean and standard deviation

    Parameters:
        n (float): shape parameter used in Negative Binomial
        p (float): shape parameter used in Negative Binomial
    
    Returns:
        mean (float): mean value of mixture model
        stdev (float): standard deviation of mixture model

    """
    mean = (n) * ((1 - p)/(p))
    stdev = (n) * ((1 - p)/(p**2))

    return mean, stdev

def _conv_np_mode(n: float,
                  p: float) -> float:
    """
    Converts Negative Binomial n, p parameters into mode

    Parameters:
        n (float): shape parameter used in Negative Binomial
        p (float): shape parameter used in Negative Binomial
    
    Returns:
        mode (float): mode of mixture model
    """
    if n <= 1:
        mode = 0
    elif n > 1:
        mode = (((n - 1)*(1 - p))/(p))

    return mode

def _find_background_comp(fit_results: dict) -> int:
    """
    For an antibody, find the background component in a mixture model by
    using the smallest mean

    Parameters:
        fit_results (dict): optimization results for an antibody, containing
            three mixture models and their respective parameter results
            Example: 
                {3: {},
                 2: {},
                 1: {}}

    Returns:
        background_component (int): index of the background component in 
            a mixture model.
            Example:
                0 = 1st component
                1 = 2nd component
                2 = 3rd component
    """
    
    best_num_comp = next(iter(fit_results.items()))[0]
    mix_model = next(iter(fit_results.items()))[1]

    # Check if results is from Negative Binomial or Gaussian 
    if mix_model['model'] == 'negative binomial (MLE)':
        component_means_nb = []

        n_params = mix_model['nb_n_params']
        p_params = mix_model['nb_p_params']

        # Calculate means for each individual component
        for comp in range (best_num_comp):
            mean = ss.nbinom.mean(n_params[comp], p_params[comp])
            component_means_nb.append(mean)

        # Return component with the smallest mean
        background_component = component_means_nb.index(min(component_means_nb))

    elif mix_model['model'] == 'gaussian (EM)':
        gmm_means = mix_model['gmm_means']
        background_component = gmm_means.index(min(gmm_means))
        
    return background_component

def _classify_cells(fit_results: dict,
                    data_vector: list, 
                    bg_comp: int, 
                    epsilon: float = 0.5) -> list:
    """
    Classifies cells as either background or signal for a single antibody

    Parameters:
        fit_results (dict): optimization results for an antibody, containing
            three mixture models and their respective parameter results
        data_vector (list): raw counts from protein data
        bg_comp (int): integer representing background component
        epsilon (float): adjustable value added to probabilities of signal cells
    
    Returns:
        classified_cells (list): list of cells, with "0" indicating background
            cell and "1" indicating signal cell
    """
    classified_cells = []

    mix_model = next(iter(fit_results.items()))[1]
    best_num_mix = next(iter(fit_results.items()))[0]

    component_list = [x for x in range(best_num_mix)] # ex: [0, 1] or [0, 1, 2]

    if mix_model['model'] == 'negative binomial (MLE)':
        n_params = mix_model['nb_n_params']
        p_params = mix_model['nb_p_params']

        # Given background component, find its associated parameters
        bg_n = n_params[bg_comp]
        bg_p = p_params[bg_comp]

        component_list.remove(bg_comp)

        # comp1 will be the background component
        comp1_probs = ss.nbinom.pmf(range(max(data_vector) + 1), 
                                    bg_n, 
                                    bg_p)

        # find the mode of the fitted background component
        comp1_mode = _conv_np_mode(bg_n, bg_p)

        if best_num_mix == 3:
            comp2_index = component_list.pop(0)
            comp2_probs = ss.nbinom.pmf(range(max(data_vector) + 1),
                                        n_params[comp2_index], 
                                        p_params[comp2_index])

            comp3_index = component_list.pop(0)
            comp3_probs = ss.nbinom.pmf(range(max(data_vector) + 1), 
                                        n_params[comp3_index], 
                                        p_params[comp3_index])

        elif best_num_mix == 2:
            comp2_index = component_list.pop(0)
            comp2_probs = ss.nbinom.pmf(range(max(data_vector) + 1), 
                                        n_params[comp2_index], 
                                        p_params[comp2_index])
        
        # Iterate over each cell value for an antibody
        for index, cell in data_vector.items():
            #print(cell)
            if best_num_mix == 3:
                comp1_cell_prob = comp1_probs[cell]
                comp2_cell_prob = comp2_probs[cell] + (0.5 - epsilon)
                comp3_cell_prob = comp3_probs[cell] + (0.5 - epsilon)

                cell_probs = [comp1_cell_prob, comp2_cell_prob, comp3_cell_prob]
                
                if max(cell_probs) == comp1_cell_prob or cell < comp1_mode:
                    classified_cells.append(0)
                else:
                    classified_cells.append(1)

            elif best_num_mix == 2:
                comp1_cell_prob = comp1_probs[cell]
                comp2_cell_prob = comp2_probs[cell] + (0.5 - epsilon)

                cell_probs = [comp1_cell_prob, comp2_cell_prob]
                
                if max(cell_probs) == comp1_cell_prob or cell < comp1_mode:
                    classified_cells.append(0)
                else:
                    classified_cells.append(1)

    elif mix_model['model'] == 'gaussian (EM)':
        gmm_means = mix_model['gmm_means']
        gmm_stdevs = mix_model['gmm_stdevs']

        bg_mean = gmm_means[bg_comp]
        bg_stdev = gmm_stdevs[bg_comp]

        component_list.remove(bg_comp)
        
        # mode of background will be equal to the mean
        comp1_mode = bg_mean

        if best_num_mix == 3:
            comp2_index = component_list.pop(0)
            comp2_mean = gmm_means[comp2_index]
            comp2_stdev = gmm_stdevs[comp2_index]
            
            comp3_index = component_list.pop(0)
            comp3_mean = gmm_means[comp3_index]
            comp3_stdev = gmm_stdevs[comp3_index]
           
        elif best_num_mix == 2:
            comp2_index = component_list.pop(0)
            comp2_mean = gmm_means[comp2_index]
            comp2_stdev = gmm_stdevs[comp2_index]
            
        # Iterate over each cell value for an antibody
        for index, cell in data_vector.items():
            if best_num_mix == 3:
                comp1_cell_prob = ss.norm.pdf(cell,
                                              bg_mean,
                                              bg_stdev)
                comp2_cell_prob = ss.norm.pdf(cell,
                                              comp2_mean,
                                              comp2_stdev) + (0.5 - epsilon)
                comp3_cell_prob = ss.norm.pdf(cell,
                                              comp3_mean,
                                              comp3_stdev) + (0.5 - epsilon)

                cell_probs = [comp1_cell_prob, comp2_cell_prob, comp3_cell_prob]
                
                if max(cell_probs) == comp1_cell_prob or cell < comp1_mode:
                    classified_cells.append(0)
                else:
                    classified_cells.append(1)

            elif best_num_mix == 2:
                comp1_cell_prob = ss.norm.pdf(cell,
                                              bg_mean,
                                              bg_stdev)
                comp2_cell_prob = ss.norm.pdf(cell,
                                              comp2_mean,
                                              comp2_stdev) + (0.5 - epsilon)
                cell_probs = [comp1_cell_prob, comp2_cell_prob]
                
                if max(cell_probs) == comp1_cell_prob or cell < comp1_mode:
                    classified_cells.append(0)
                else:
                    classified_cells.append(1)

    return classified_cells

def _classify_cells_df(fit_all_results: list, 
                       protein_data: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
        fit_all_results (list): optimization results for all antibodies
        protein_data (pd.DataFrame): count matrix containing antibodies x cells
    
    Returns:
        classified_df (pd.DataFrame): Pandas DataFrame containing "0" and "1",
            where "0" represents a background expression of a cell for a given
            antibody, and "1" represents a antibody-specific signal expression
    """

    # Storing antibody name with each classified vector of cells
    count_matrix_columns = list(protein_data.columns)
    count_matrix_index = list(protein_data.index)
    
    classified_matrix = []

    for index, ab in enumerate(protein_data):
        # Find background component
        ab_background_comp_index = _find_background_comp(fit_all_results[index])

        # Classify cells in vector as either background or signal
        ab_classified_cells = _classify_cells(fit_all_results[index], 
                                              protein_data.loc[:, ab], 
                                              ab_background_comp_index)

        classified_matrix.append(ab_classified_cells)

    # Transpose matrix:
    classified_transpose = list(map(list, zip(*classified_matrix)))

    classified_df = pd.DataFrame(classified_transpose, 
                                 index=count_matrix_index, 
                                 columns=count_matrix_columns)

    return classified_df