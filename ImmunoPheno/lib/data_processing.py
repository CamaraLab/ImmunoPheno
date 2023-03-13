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