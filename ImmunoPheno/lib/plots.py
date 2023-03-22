import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import seaborn as sns
import plotly.express as px
import umap

import data_processing
 
def plot_fits(counts: list,
              fit_results: dict,
              plot_percentile: float = 99.5,
              transformed: bool = False):
    """
    Plots the fits of the mixture model for an antibody

    Parameters:
        counts (list): count values of cells for an antibody
        fit_results (dict): optimization results of a mixture model
        plot_percentile (float): plot range of graph based on percentile
        transformed (bool): indicate whether data has been transformed 
    
    Returns:
        Matplotlib graph containing a histogram of the counts and the plots
        containing the individual components of each model
    """
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 5))

    plot_min = math.floor(np.percentile(counts, 100-plot_percentile))
    plot_max = math.ceil(np.percentile(counts, plot_percentile))
    
    # Change plot range if data is transformed
    if transformed:
        plot_range = np.arange(plot_min, plot_max, 0.01)
    else:
        if plot_max < 100:
            plot_max = 100
        plot_range = range(plot_min, plot_max)


    # Get results from fit_results dict
    for key, value in sorted(fit_results.items()):
        if value['model'] == 'gaussian (EM)':
            # Get all parameters:
            gmm_thetas = value['gmm_thetas']
            gmm_means = value['gmm_means']
            gmm_stdevs = value['gmm_stdevs']

            if value['num_comp'] == 1:
                ax[0].set_title("Gaussian Mixture Model (EM): 1 component")
                ax[0].hist(counts, 
                            bins=100, 
                            range=[plot_min, plot_max], 
                            density=True,
                            alpha=0.5)

                ax[0].plot(plot_range,
                            gmm_thetas[0] * (ss.norm.pdf(plot_range, 
                                                        gmm_means[0], 
                                                        gmm_stdevs[0])))
            
            if value['num_comp'] == 2:
                ax[1].set_title("Gaussian Mixture Model (EM): 2 component")
                ax[1].hist(counts, 
                           bins=100, 
                           range=[plot_min, plot_max], 
                           density=True, 
                           alpha=0.5)
                
                # Combined components
                ax[1].plot(plot_range, 
                           gmm_thetas[0] * (ss.norm.pdf(plot_range, 
                                                        gmm_means[0], 
                                                        gmm_stdevs[0])) 
                         + ((1 - gmm_thetas[0]) * (ss.norm.pdf(plot_range, 
                                                               gmm_means[1], 
                                                               gmm_stdevs[1]))))

                # Component 1
                ax[1].plot(plot_range, 
                           gmm_thetas[0] * (ss.norm.pdf(plot_range, 
                                                        gmm_means[0], 
                                                        gmm_stdevs[0])),'g-')
                
                # Component 2
                ax[1].plot(plot_range, 
                           ((1 - gmm_thetas[0]) 
                         * (ss.norm.pdf(plot_range, 
                                        gmm_means[1], 
                                        gmm_stdevs[1]))),'r-')
                
            if value['num_comp'] == 3:
                ax[2].set_title("Gaussian Mixture Model (EM): 3 component")
                ax[2].hist(counts, 
                           bins=100, 
                           range=[plot_min, plot_max], 
                           density=True, 
                           alpha=0.5)
                
                # Combined components
                ax[2].plot(plot_range, 
                           gmm_thetas[0] * (ss.norm.pdf(plot_range,
                                                        gmm_means[0],
                                                        gmm_stdevs[0]))
                         + gmm_thetas[1] * (ss.norm.pdf(plot_range,
                                                        gmm_means[1], 
                                                        gmm_stdevs[1]))
                         + (1 - gmm_thetas[0] - gmm_thetas[1])
                         * (ss.norm.pdf(plot_range, 
                                        gmm_means[2], 
                                        gmm_stdevs[2])))

                # Component 1
                ax[2].plot(plot_range, 
                           gmm_thetas[0] * (ss.norm.pdf(plot_range,
                                                        gmm_means[0],
                                                        gmm_stdevs[0])),'g-')
                
                # Component 2
                ax[2].plot(plot_range, 
                            gmm_thetas[1] * (ss.norm.pdf(plot_range,
                                                         gmm_means[1], 
                                                         gmm_stdevs[1])),'r-')
                
                # Component 3
                ax[2].plot(plot_range, 
                           (1 - gmm_thetas[0] - gmm_thetas[1])
                         * (ss.norm.pdf(plot_range, 
                                        gmm_means[2], 
                                        gmm_stdevs[2])), 'b-')

            
                
            
                plt.show()

        elif value['model'] == 'negative binomial (MLE)':
            # Get all parameters
            nb_thetas = value['nb_thetas']
            nb_n_params = value['nb_n_params']
            nb_p_params = value['nb_p_params']
            
            if value['num_comp'] == 1:
                ax[0].set_title(("Negative Binomial " 
                                    "Mixture Model (MLE): 1 component"))
                ax[0].hist(counts, 
                        bins=100, 
                        range=[0, plot_max], 
                        density=True, 
                        alpha=0.5)

                ax[0].plot(range(plot_max), 
                        (ss.nbinom.pmf(range(plot_max), 
                                        nb_n_params[0], 
                                        nb_p_params[0])))
            
            if value['num_comp'] == 2:
                ax[1].set_title(("Negative Binomial " 
                                 "Mixture Model (MLE): 2 component"))
                ax[1].hist(counts, 
                        bins=100,
                        range=[0, plot_max], 
                        density=True, 
                        alpha=0.5)
                
                ax[1].plot(range(plot_max), 
                        nb_thetas[0] 
                        * (ss.nbinom.pmf(range(plot_max), 
                                         nb_n_params[0], 
                                         nb_p_params[0])) 
                        + (1 - nb_thetas[0])
                        * (ss.nbinom.pmf(range(plot_max), 
                                         nb_n_params[1], 
                                         nb_p_params[1]))) 
                
                ax[1].plot(range(plot_max), 
                        nb_thetas[0] 
                        * (ss.nbinom.pmf(range(plot_max), 
                                         nb_n_params[0], 
                                         nb_p_params[0])), 'g-')
                
                ax[1].plot(range(plot_max), 
                        (1 - nb_thetas[0])
                        * (ss.nbinom.pmf(range(plot_max), 
                                         nb_n_params[1], 
                                         nb_p_params[1])), 'r-')
            
            if value['num_comp'] == 3:
                ax[2].set_title(("Negative Binomial " 
                                 "Mixture Model (MLE): 3 component"))
                ax[2].hist(counts, 
                        bins=100, 
                        range=[0, plot_max], 
                        density=True, 
                        alpha=0.5)
                
                ax[2].plot(range(plot_max), 
                        nb_thetas[0] 
                        * (ss.nbinom.pmf(range(plot_max),
                                         nb_n_params[0], 
                                         nb_p_params[0])) 
                        + nb_thetas[1] 
                        * (ss.nbinom.pmf(range(plot_max),
                                         nb_n_params[1], 
                                         nb_p_params[1])) 
                        + (1 - nb_thetas[0] 
                             - nb_thetas[1]) 
                        * (ss.nbinom.pmf(range(plot_max), 
                                         nb_n_params[2], 
                                         nb_p_params[2])))

                ax[2].plot(range(plot_max), 
                        nb_thetas[0] 
                        * (ss.nbinom.pmf(range(plot_max),
                                         nb_n_params[0], 
                                         nb_p_params[0])), 'g-')  
                ax[2].plot(range(plot_max), 
                        nb_thetas[1] 
                        * (ss.nbinom.pmf(range(plot_max),
                                         nb_n_params[1], 
                                         nb_p_params[1])), 'r-')  
                ax[2].plot(range(plot_max), 
                        (1 - nb_thetas[0] 
                           - nb_thetas[1]) 
                        * (ss.nbinom.pmf(range(plot_max), 
                                         nb_n_params[2], 
                                         nb_p_params[2])), 'b-')

                plt.show()

def plot_all_fits(protein_data: pd.DataFrame,
                  all_fits: list,
                  plot_percentile: float = 99.5,
                  transformed: bool = False):
    """
    Plots all antibody histograms and mixture model fits

    Parameters:
        protein_data (pd.DataFrame): matrix containing cells x antibodies
        all_fits (list): list of dictionaries, with each dictionary 
            containing results from optimization for an antibody 
        plot_percentile (float): set plot range of graph based on percentile
        transformed (bool): indicate whether data has been transformed
    
    Returns:
        A series of plots with each type of mixture model for every antibody
        in the protein data
    """
    
    for index, ab in enumerate(protein_data):
        print("Antibody:", ab)
        plot_fits(protein_data.loc[:, ab], 
                  all_fits[index],
                  plot_percentile=plot_percentile,
                  transformed=transformed)

def plot_antibody_correlation(IPD):
    """
    Plots a correlation heatmap for each antibody in the data

    Parameters:
        IPD (ImmunoPhenoData Object): Object containing protein data,
            gene data, and cell types
    
    Returns:
        seaborn clustermap for a heatmap of the antibodies
    """
    # Calculate correlation dataframe
    corr_df = data_processing._correlation_ab(IPD.classified_filt, IPD.z_scores)
    g = sns.clustermap(corr_df, vmin=-1, vmax=1, cmap='BrBG')
    g.ax_cbar.set_position((1, .2, .03, .4))

def plot_UMAP(IPD,
              normalized: bool = False):
    """ 
    Plots a UMAP for the non-normalized protein values or normalized protein
    values

    Parameters:
        IPD (ImmunoPhenoData Object): Object containing protein data,
            gene data, and cell types
        normalized (bool): option to plot normalized values

    Returns:
        UMAP projection of non/normalized protein values with a corresponding
        legend of cell type (if available)
    """
    umap_plot = umap.UMAP(random_state=0)
    if normalized:
        try:
            projections = umap_plot.fit_transform(IPD.normalized_counts)
        except:
            raise data_processing.PlotUMAPError("Cannot plot normalized UMAP. "
                            "normalize_all_antibodies() must be called first.")
    else:
        projections = umap_plot.fit_transform(IPD.protein_cleaned)
    
    if IPD.raw_cell_labels is None:
        norm_plot = px.scatter(
            projections, x=0, y=1,
        )
        norm_plot.show()
    elif IPD.raw_cell_labels is not None:
        if normalized:
            raw_types = IPD.norm_cell_labels.iloc[:, 0].tolist()
            norm_plot = px.scatter(
                projections, x=0, y=1,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                color=[str(cell_type) for cell_type in raw_types],
                labels={'color':'cell type'}
            )
            norm_plot.show()
                
        else:
            raw_types = IPD.raw_cell_labels.iloc[:, 0].tolist()
            reg_plot = px.scatter(
                projections, x=0, y=1,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                color=[str(cell_type) for cell_type in raw_types],
                labels={'color':'cell type'}
            )
            reg_plot.show()