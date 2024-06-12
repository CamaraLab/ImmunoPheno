import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as ss
import scipy.optimize as so
import scipy.special as sp

from sklearn.mixture import GaussianMixture
import plotly.express as px
from plotly.subplots import make_subplots

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

def _plot_fits(counts: list,
              fit_results: dict,
              ab_name: str,
              width:int = None):
    """
    Plots the fits of the mixture model for an antibody

    Parameters:
        counts (list): count values of cells for an antibody
        fit_results (dict): optimization results of a mixture model
        ab_name (str): name of antibody
        width (int): display width of all the graphs (optional)
    
    Returns:
        Plotly graph containing a histogram of the counts and the plots
        containing the individual components of each model
    """
    
    # Make initial subplots 
    fig = make_subplots(rows=1, 
                        cols=3, 
                        subplot_titles=("Negative Binomial Mixture Model (MLE): 1 Component",
                                        "Negative Binomial Mixture Model (MLE): 2 Components", 
                                        "Negative Binomial Mixture Model (MLE): 3 Components"))
    
    traceHistogram = px.histogram(counts, 
                                  histnorm='probability density', 
                                  opacity=0.75, 
                                  color_discrete_sequence=['#1F77B4'])
    
    fig.add_trace(traceHistogram.data[0], row=1, col=1)
    fig.add_trace(traceHistogram.data[0], row=1, col=2)
    fig.add_trace(traceHistogram.data[0], row=1, col=3)
    
    # Find best mixture model
    best_mixture_model = next(iter(fit_results.items()))[0]
    
    # AICs in order of 1 mixture, 2 mixtures, 3 mixtures
    aics = []

    # Adjusted Y axis max
    y_axis_max = 0
        
    for key, value in sorted(fit_results.items()):
        if value["model"] == 'negative binomial (MLE)':
            # Find background component for each mixture model
            tempDict = {key: value}
            background_component =  _find_background_comp(tempDict)
            
            # Get all parameters
            nb_thetas = value["nb_thetas"]
            nb_n_params = value["nb_n_params"]
            nb_p_params = value["nb_p_params"]
            aic = value["aic"]
            aics.append(aic)

            x_values = range(int(min(counts)), int(max(counts)))
            
            if value['num_comp'] == 1:
                # Single component will always be background, so make it red
                trace1a = px.line(x=x_values,
                                  y=1 * (ss.nbinom.pmf(x_values, 
                                                       nb_n_params[0], 
                                                       nb_p_params[0])), 
                                  color_discrete_sequence=["red"])

                fig.add_trace(trace1a.data[0], row=1, col=1)
            
            if value["num_comp"] == 2:
                # Find a weighted average of each component maximum for Y axis scale
                component_maximums = []
                first_max = max(nb_thetas[0]
                                 * (ss.nbinom.pmf(x_values,
                                                  nb_n_params[0],
                                                  nb_p_params[0])))
                second_max = max((1 - nb_thetas[0])
                                 * (ss.nbinom.pmf(x_values,
                                                  nb_n_params[1],
                                                  nb_p_params[1])))
                component_maximums.append(first_max)
                component_maximums.append(second_max)

                # Weighted average will bias towards the smaller of the two maximums
                y_axis_max = ((2 * min(component_maximums)) + max(component_maximums)) / 3
                
                # Plot combined components
                trace2 = px.line(x=x_values,
                                 y=nb_thetas[0]
                                 * (ss.nbinom.pmf(x_values,
                                                  nb_n_params[0],
                                                  nb_p_params[0]))
                                 + (1 - nb_thetas[0])
                                 * (ss.nbinom.pmf(x_values,
                                                  nb_n_params[1],
                                                  nb_p_params[1])),
                                color_discrete_sequence=["orange"])
                fig.add_trace(trace2.data[0], row=1, col=2)
                
                # Plotting individual components
                line_colors = ["green"]
                for index in range(2):
                    tempThetas = nb_thetas.copy()
                    tempThetas.append(1 - nb_thetas[0])
                    
                    # If this is the background component
                    if index == background_component:
                        trace2a = px.line(x=x_values,
                                          y=tempThetas[index] 
                                          * (ss.nbinom.pmf(x_values, 
                                                           nb_n_params[index], 
                                                           nb_p_params[index])), 
                                            color_discrete_sequence=["red"])
                        fig.add_trace(trace2a.data[0], row=1, col=2)
                    else:
                        nextColor = line_colors.pop()
                        trace2b = px.line(x=x_values,
                                          y=tempThetas[index] 
                                          * (ss.nbinom.pmf(x_values, 
                                                           nb_n_params[index], 
                                                           nb_p_params[index])), 
                                            color_discrete_sequence=[nextColor])
                        fig.add_trace(trace2b.data[0], row=1, col=2)
            
            if value["num_comp"] == 3:
                # Plot combined components
                trace3 = px.line(x=x_values,
                                 y=nb_thetas[0]
                                 * (ss.nbinom.pmf(x_values,
                                                  nb_n_params[0],
                                                  nb_p_params[0]))
                                 + nb_thetas[1]
                                 * (ss.nbinom.pmf(x_values,
                                                  nb_n_params[1],
                                                  nb_p_params[1]))
                                 + (1 - nb_thetas[0]
                                      - nb_thetas[1])
                                 * (ss.nbinom.pmf(x_values,
                                                  nb_n_params[2],
                                                  nb_p_params[2])),
                                    color_discrete_sequence=["orange"])
                fig.add_trace(trace3.data[0], row=1, col=3)
                
                # Plotting individual components
                line_colors = ["blue", "green"]
                
                for index in range(3):
                    tempThetas = nb_thetas.copy()
                    tempThetas.append(1 - nb_thetas[0] - nb_thetas[1])
                    
                    # If this is the background component
                    if index == background_component:
                        trace3a = px.line(x=x_values,
                                          y=tempThetas[index] 
                                          * (ss.nbinom.pmf(x_values, 
                                                           nb_n_params[index], 
                                                           nb_p_params[index])), 
                                            color_discrete_sequence=["red"])
                        fig.add_trace(trace3a.data[0], row=1, col=3)
                    else:
                        nextColor = line_colors.pop()
                        trace3b = px.line(x=x_values,
                                          y=tempThetas[index] 
                                          * (ss.nbinom.pmf(x_values, 
                                                           nb_n_params[index], 
                                                           nb_p_params[index])), 
                                            color_discrete_sequence=[nextColor])
                        fig.add_trace(trace3b.data[0], row=1, col=3)
                        
        elif value["model"] == "gaussian (EM)":
            # Find background component for each mixture model
            tempDict = {key: value}
            background_component =  _find_background_comp(tempDict)
            
            # Get all parameters:
            gmm_thetas = value['gmm_thetas']
            gmm_means = value['gmm_means']
            gmm_stdevs = value['gmm_stdevs']
            aic = value["aic"]
            aics.append(aic)

            x_values = np.linspace(min(counts), max(counts), len(counts))
            
            if value["num_comp"] == 1:
                # Update title to Gaussian
                fig.layout.annotations[0].update(text="Gaussian Mixture Model (EM): 1 Component")
                
                 # Single component will always be background, so make it red
                trace1a = px.line(x=x_values, 
                                  y=1 * (ss.norm.pdf(x_values, 
                                                     gmm_means[0], 
                                                     gmm_stdevs[0])), 
                                  color_discrete_sequence=["red"])

                fig.add_trace(trace1a.data[0], row=1, col=1)
            
            if value["num_comp"] == 2:
                # Find a weighted average of each component maximum for Y axis scale
                component_maximums = []
                first_max = max(gmm_thetas[0]
                                 * (ss.norm.pdf(x_values,
                                                gmm_means[0],
                                                gmm_stdevs[0])))
                second_max = max((1 - gmm_thetas[0])
                                 * (ss.norm.pdf(x_values,
                                                gmm_means[1],
                                                gmm_stdevs[1])))
                component_maximums.append(first_max)
                component_maximums.append(second_max)

                # Weighted average will bias towards the smaller of the two maximums
                y_axis_max = ((2 * min(component_maximums)) + max(component_maximums)) / 3
                
                # Update title to Gaussian
                fig.layout.annotations[1].update(text="Gaussian Mixture Model (EM): 2 Components")
                
                # Plot combined components
                trace2 = px.line(x=x_values,
                                 y=gmm_thetas[0]
                                 * (ss.norm.pdf(x_values,
                                                gmm_means[0],
                                                gmm_stdevs[0]))
                                 + (1 - gmm_thetas[0])
                                 * (ss.norm.pdf(x_values,
                                                gmm_means[1],
                                                gmm_stdevs[1])),
                                color_discrete_sequence=["orange"])
                
                fig.add_trace(trace2.data[0], row=1, col=2)
                
                # Plotting individual components
                line_colors = ["green"]
                for index in range(2):
                    tempThetas = gmm_thetas.copy()
                    tempThetas.append(1 - gmm_thetas[0])
                    
                    # If this is the background component
                    if index == background_component:
                        trace2a = px.line(x=x_values,
                                          y=tempThetas[index] 
                                          * (ss.norm.pdf(x_values, 
                                                         gmm_means[index], 
                                                         gmm_stdevs[index])), 
                                            color_discrete_sequence=["red"])
                        fig.add_trace(trace2a.data[0], row=1, col=2)
                    else:
                        nextColor = line_colors.pop()
                        trace2b = px.line(x=x_values,
                                          y=tempThetas[index] 
                                          * (ss.norm.pdf(x_values, 
                                                         gmm_means[index], 
                                                         gmm_stdevs[index])), 
                                            color_discrete_sequence=[nextColor])
                        fig.add_trace(trace2b.data[0], row=1, col=2)
                
            if value["num_comp"] == 3:
                # Update title to Gaussian
                fig.layout.annotations[2].update(text="Gaussian Mixture Model (EM): 3 Components")
                
                # Plot combined components
                trace3 = px.line(x=x_values,
                                 y=gmm_thetas[0]
                                 * (ss.norm.pdf(x_values,
                                                gmm_means[0],
                                                gmm_stdevs[0]))
                                 + gmm_thetas[1]
                                 * (ss.norm.pdf(x_values,
                                                gmm_means[1],
                                                gmm_stdevs[1]))
                                 + (1 - gmm_thetas[0]
                                      - gmm_thetas[1])
                                 * (ss.norm.pdf(x_values,
                                                gmm_means[2],
                                                gmm_stdevs[2])),
                                    color_discrete_sequence=["orange"])
                fig.add_trace(trace3.data[0], row=1, col=3)
                
                # Plotting individual components
                line_colors = ["blue", "green"]
                
                for index in range(3):
                    tempThetas = gmm_thetas.copy()
                    tempThetas.append(1 - gmm_thetas[0] - gmm_thetas[1])
                    
                    # If this is the background component
                    if index == background_component:
                        trace3a = px.line(x=x_values,
                                          y=tempThetas[index] 
                                          * (ss.norm.pdf(x_values, 
                                                         gmm_means[index], 
                                                         gmm_stdevs[index])), 
                                            color_discrete_sequence=["red"])
                        fig.add_trace(trace3a.data[0], row=1, col=3)
                    else:
                        nextColor = line_colors.pop()
                        trace3b = px.line(x=x_values,
                                          y=tempThetas[index] 
                                          * (ss.norm.pdf(x_values, 
                                                         gmm_means[index], 
                                                         gmm_stdevs[index])), 
                                            color_discrete_sequence=[nextColor])
                        fig.add_trace(trace3b.data[0], row=1, col=3)
                
    # Add "AIC: " prefix and bold heading to list of AICs
    formatted_aics = []
    for count, aic in enumerate(aics):
        if count == best_mixture_model - 1:
            formatted_aics.append(f"<b>AIC: {aic}<b>")
        else:
            formatted_aics.append(f"AIC: {aic}")
    
    fig.for_each_annotation(lambda a: a.update(text=f'{a.text}'))
    fig.update_annotations(font=dict(family="Calibri", size=15))
    fig.update_layout(height=600,
                      width=width,
                      title_text=f'Antibody: {ab_name}' +'<br>' + f'<span style="font-size: 15px;">Best model: <b>{best_mixture_model} Component <b></span>', 
                      showlegend=False)
                  
    fig.update_yaxes(range=[0, y_axis_max], row=1, col=1)
    fig.update_yaxes(range=[0, y_axis_max], row=1, col=2)
    fig.update_yaxes(range=[0, y_axis_max], row=1, col=3)

    fig['layout']['xaxis']['title']= formatted_aics[0]
    fig['layout']['xaxis2']['title']= formatted_aics[1]
    fig['layout']['xaxis3']['title']= formatted_aics[2]
    
    fig.show()

def plot_ab_fit(IPD, ab: str, width: int = None):
    """
    Plots an antibody histogram and mixture model fits

    Parameters:
        IPD (ImmunoPhenoData Object): Object containing all fits from 
            mixture models
        ab (str): name of antibody to retrieve fits 
        width (int): display width of all the graphs (optional)

    Returns:
        None. Renders a series of plots (1, 2, 3 component mixture models) for an
        antibody after fitting with a negative binomial or gaussian curve.
    """
    if IPD._all_fits is None:
        raise Exception("No fits found. Call fit_all_antibodies first")
    
    if len(IPD._all_fits_dict) == 0:
        raise Exception("No fits found. Call fit_all_antibodies first")

    if isinstance(ab, str):
        try:
            _plot_fits(counts=IPD.protein.loc[:, ab],
                      fit_results=IPD._all_fits_dict[ab],
                      ab_name=ab,
                      width=width)
        except:
            raise Exception(f"'{ab}' not found in protein data.")

def plot_all_fits(IPD, width: int = None):
    """
    Plots all antibody histograms and mixture model fits

    Parameters:
        IPD (ImmunoPhenoData Object): Object containing all fits from 
            mixture models
        width (int): display width of all the graphs (optional)
    
    Returns:
        None. Renders a series of plots with each type of mixture model for every antibody
        in the protein data
    """
    if IPD._all_fits is None:
        raise Exception("No fits found. Call fit_all_antibodies first")
    
    if len(IPD._all_fits_dict) == 0:
        raise Exception("No fits found. Call fit_all_antibodies first")
    
    for index, ab in enumerate(IPD.protein):
        _plot_fits(counts=IPD.protein.loc[:, ab],
                  fit_results=IPD._all_fits_dict[ab],
                  ab_name=ab,
                  width=width)

def _gmm_init_params(counts: list,
                     n_components: int,
                     random_state: int = 0,
                     **kwargs) -> tuple:
    """
    Generates parameters after fitting a Gaussian Mixture Model

    Parameters:
        counts (list): raw counts from protein data
        n_components (int): number of components for mixture model
        random_state (int): seed value
        kwargs: initial arguments for sklearn's GaussianMixture (optional)

    Returns:
        gmm_means (list): mean values of each mixture component in a model
        gmm_stdevs (list): standard deviations of each mixture component
        gmm_thetas (list): weights of each mixture components
        gmm_log_like (float): log-likelihood of the model
        gmm_aic (float): Akaike information criterion of the model
    """

    # Convert counts into numpy array and reshape (due to GMM input formatting)
    counts_reshaped = np.array(counts).reshape(-1, 1)

    gmm_params = GaussianMixture(n_components=n_components,
                                 random_state=random_state,
                                 **kwargs).fit(counts_reshaped)

    gmm_means = gmm_params.means_.flatten().tolist()
    gmm_variances = gmm_params.covariances_.flatten().tolist()
    gmm_stdevs = np.sqrt(gmm_variances).tolist()
    gmm_thetas = gmm_params.weights_.flatten().tolist()
    
    gmm_log_like = gmm_params.score_samples(X=counts_reshaped).sum()
    gmm_aic = gmm_params.aic(X=counts_reshaped)

    return gmm_means, gmm_stdevs, gmm_thetas, gmm_log_like, gmm_aic

def _gmm_results(counts: list,
                 ab_name: str,
                 plot: bool = False, 
                 random_state: int = 0,
                 **kwargs) -> dict:
    """
    Generates the GMM results for 1, 2, and 3 component mixture models
    Plots the fit of each model on a histogram of the data

    Parameters:
        counts (list): raw counts from protein data
        plot (bool): option to plot each model
        random_state (int): seed value
        **kwargs: initial arguments for sklearn's GaussianMixture (optional)

    Returns:
        sorted_results (dict): all models and their parameters, sorted by
                               the AIC values in ascending order
        Plots will be printed if plot (bool) is set to True
    """
    results = {}

    for num_components in range (1, 4):
        gmm_means, gmm_stdevs, gmm_thetas, \
        gmm_log_like, gmm_aic = _gmm_init_params(counts=counts, 
                                                 n_components=num_components, 
                                                 random_state=random_state,
                                                 **kwargs)
        
        results[num_components] = {
            'num_comp': num_components,
            'model': 'gaussian (EM)',
            'aic': gmm_aic,
            'gmm_means': gmm_means,
            'gmm_stdevs': gmm_stdevs,
            'gmm_thetas': gmm_thetas,
            'gmm_log_like': gmm_log_like,
        }

    sorted_results = dict(sorted(results.items(), 
                                 key=lambda item: item[1]['aic']))

    if plot:
        _plot_fits(counts=counts,
                  fit_results=sorted_results,
                  ab_name=ab_name)
        
    return sorted_results

def _convert_gmm_np(mean: float, 
                    stdev: float) -> tuple:
    """
    Converts parameters from GMM to n, p parameterization for
    usage in Negative Binomial Mixture Models

    Parameters:
        mean (float): mean value of a mixture component
        stdev (float): standard deviation of a mixture component

    Returns:
        n_param (float): shape parameter used in a Negative Binomial Model
        p_param (float): shape parameter used in a Negative Binomial Model
    """
    if mean == 0:
        mean += 1e-8
    
    # If stdev is less than mean, convert to Poisson distribution
    if (stdev ** 2) < mean:
        variance = mean + 0.00001
    else:
        variance = stdev**2
    
    n_param = (mean**2) / (variance - mean)
    p_param =  (variance - mean) / variance
    
    return n_param, (1 - p_param)

def _init_params_np(gmm_means: list, 
                    gmm_stdevs: list, 
                    gmm_thetas: list) -> tuple:
    """
    Generate initial parameters for a Negative Binomial Mixture Model 
    using the results from a Gaussian Mixture Model 
    
    Parameters:
        gmm_means (array): mean values generated from GMM
        gmm_stdevs (array): standard deviations generated from GMM
        gmm_thetas (arary): weights generated from GMM

    Returns:
        init_params (list): list of parameters ordered by units of [n, p, theta]
            depending on the number of components. 
            Example: 2 components [n1, p1, theta1, n2, p2, theta2]
        num_components (int): the number of components in the model
    """
    init_params = []

    # Find number of components in model
    num_components = len(gmm_thetas)

    if num_components == 1:
        temp_n, temp_p = _convert_gmm_np(gmm_means[0], gmm_stdevs[0])
        init_params.append(temp_n)
        init_params.append(temp_p)

    elif num_components == 2:
        for i in range(num_components):
            temp_n, temp_p = _convert_gmm_np(gmm_means[i], gmm_stdevs[i])
            init_params.append(temp_n)
            init_params.append(temp_p)
        
        # Add first theta 
        init_params.append(gmm_thetas[0])

    elif num_components == 3:
        for i in range(num_components):
            temp_n, temp_p = _convert_gmm_np(gmm_means[i], gmm_stdevs[i])
            init_params.append(temp_n)
            init_params.append(temp_p)

        # Add first two thetas
        for i in range(num_components - 1):
            init_params.append(gmm_thetas[i])

    return init_params, num_components

def _param_bounds(num_components: int) -> list:
    """
    Generate the appropriate bounds for a mixture model depending on 
    the number of components present. Bounds will be used during optimization.

    Parameters:
        num_components (int): number of components in a mixture model
    
    Returns:
        bounds (list): bounded value for each parameter in the list
    """
    bounds = []

    if num_components == 1:
        for i in range(2 * num_components):
            bounds.append((1e-8, np.inf))

    elif num_components == 2:
        for i in range(num_components):
            bounds.append((1e-8, np.inf))
            bounds.append((1e-8, np.inf))

        bounds.append((1e-8, 1))

    elif num_components == 3:
        for i in range(num_components):
            bounds.append((1e-8, np.inf))
            bounds.append((1e-8, np.inf))

        for i in range(num_components - 1):
            bounds.append((1e-8, 1))
    
    return bounds

def _convert_np_ab(args: tuple,
                   num_components: int) -> tuple:
    """
    Convert n, p parameterization to alpha, b parameterization for 
    Negative Binomial maximum likelihood estimation and optimization

    Parameters:
        args (tuple): n, p parameters and weights (thetas) generated from GMM
        num_components (int): number of components in the mixture model
    
    Returns:
        alpha (float): initial guess for Negative Binomial estimation
        b (float): initial guess for Negative Binomial estimation
        theta (float): initial guess for Negative Binomial estimation
    """

    if num_components == 1:
        n1, p1 = args
        
        alpha1 = n1
        b1 = (1 - p1) / p1

        return alpha1, b1

    elif num_components == 2:
        n1, p1, n2, p2, theta = args

        alpha1 = n1
        alpha2 = n2

        b1 = (1 - p1) / p1
        b2 = (1 - p2) / p2
        
        return alpha1, b1, alpha2, b2, theta

    elif num_components == 3:
        n1, p1, n2, p2, n3, p3, theta1, theta2 = args

        alpha1 = n1
        alpha2 = n2
        alpha3 = n3

        b1 = (1 - p1) / p1
        b2 = (1 - p2) / p2
        b3 = (1 - p3) / p3

        return alpha1, b1, alpha2, b2, alpha3, b3, theta1, theta2
        
def _log_like_nb(args: tuple, 
                 data_vector: list, 
                 num_components: int) -> float:
    """
    Calculates the log-likelihood value for a Negative Binomial mixture model

    Parameters:
        args (tuple): initial alpha, b, theta parameters 
        data_vector (list): raw counts from protein data
        num_components (int): number of components in mixture model

    Returns:
        The negative sum of the log likelihoods for a mixture model
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if num_components == 1:
            alpha1, b1 = args

            logx = np.log(ss.nbinom.pmf(list(data_vector),
                                        alpha1,
                                        (1/(1 + b1))))

            return (-1) * np.sum(logx)
            
        elif num_components == 2:
            alpha1, b1, alpha2, b2, theta = args
            
            logx1 = np.log(ss.nbinom.pmf(list(data_vector),
                                         alpha1,
                                         (1/(1 + b1))))
            logx2 = np.log(ss.nbinom.pmf(list(data_vector),
                                         alpha2,
                                         (1/(1 + b2))))

            lse_coeffs = np.tile([theta, (1 - theta)], 
                                [len(data_vector), 1]).transpose()

            log_likes = sp.logsumexp(np.vstack([logx1, logx2]), 
                                     axis=0, 
                                     b=lse_coeffs)

            return (-1) * np.sum(log_likes)

        elif num_components == 3:
            alpha1, b1, alpha2, b2, alpha3, b3, theta1, theta2 = args
            
            logx1 = np.log(ss.nbinom.pmf(list(data_vector),
                                         alpha1,
                                         (1/(1 + b1))))
            logx2 = np.log(ss.nbinom.pmf(list(data_vector),
                                         alpha2,
                                         (1/(1 + b2))))
            logx3 = np.log(ss.nbinom.pmf(list(data_vector),
                                         alpha3,
                                         (1/(1 + b3))))

            lse_coeffs = np.tile([theta1, theta2, (1 - theta1 - theta2)], 
                                [len(data_vector), 1]).transpose()

            log_likes = sp.logsumexp(np.vstack([logx1, logx2, logx3]), 
                                    axis=0, 
                                    b=lse_coeffs)

            return (-1) * np.sum(log_likes)

def _theta_constr_mle_2(args: list) -> float:
    """
    A constraint function for the weights in a 2-component 
    Negative Binomial Mixture Model 

    Parameters:
        args (list): initial parameters used for NB optimization
            Example: [n1, p1, n2, p2, theta]      
    
    Returns:
        A bounded total weight (theta) by 1 during optimization
    """
    return 1 - args[4]

def _theta_constr_mle_3(args: list) -> float:
    """
    A constraint function for the weights in a 3-component
    Negative Binomial Mixture Model

    Parameters:
        args (list): initial parameters used for NB optimization
            Example: [n1, p1, n2, p2, n3, p3, theta1, theta2]
    
    Returns:
        A bounded total weight (theta1 - theta2) by 1 during optimization
    """
    return 1 - args[6] - args[7]

def _mle_mix_nb(args: tuple,
                data_vector: list,
                param_bounds: list,
                num_components: int) -> so.OptimizeResult:
    """
    Performs a constrained minimization using the SLSQP algorithm
    for a Negative Binomial mixture model. Objective function to be minimized
    is the _log_like_nb function. Returns the maximum likelihood estimators.

    Parameters:
        args (tuple): list of initial guesses from 
        data_vector (list): raw counts from protein data
        param_bounds (list): bounds for each parameter during minimization
        num_components (int): number of components in model

    Returns:
        res (so.OptimizeResult object): optimization result containing the
            solution and the value of the objective function.
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if num_components == 1:
            res = so.minimize(
                fun=_log_like_nb,
                x0=args,
                args=(data_vector, num_components),
                bounds = param_bounds, 
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-2},
                tol=1e-9,
        )
        elif num_components == 2:
            res = so.minimize(
                fun=_log_like_nb,
                x0=args,
                args=(data_vector, num_components),
                bounds = param_bounds,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-2},
                constraints={'type': 'ineq', 'fun': _theta_constr_mle_2},
                tol=1e-9,
        )
        elif num_components == 3:
            res = so.minimize(
                fun=_log_like_nb,
                x0=args,
                args=(data_vector, num_components),
                bounds = param_bounds,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-2},
                constraints={'type': 'ineq', 'fun': _theta_constr_mle_3},   
                tol=1e-9,
            )
        return res

def _convert_ab_np(args: tuple,
                   num_components: int) -> tuple:
    """
    Convert the results of the optimization from alpha, b to a Negative Binomial
    compatible n, p parameterization.

    Parameters:
        args (tuple): optimized parameters from maximum likelihood estimation
        num_components (int): number of components in the model

    Returns:
        alpha, b parameters converted to n, p parameters, along with the 
        associated weights (thetas) for a model
    """
    if num_components == 1:
        alpha1, b1 = args
        
        n1 = alpha1
        p1 = (1/b1) / (1 + (1/b1))

        return n1, p1

    elif num_components == 2:
        alpha1, b1, alpha2, b2, theta = args

        n1 = alpha1
        n2 = alpha2 

        p1 = (1/b1) / (1 + (1/b1))
        p2 = (1/b2) / (1 + (1/b2))

        return n1, p1, n2, p2, theta

    elif num_components == 3:
        alpha1, b1, alpha2, b2, alpha3, b3, theta1, theta2 = args

        n1 = alpha1
        n2 = alpha2
        n3 = alpha3

        p1 = (1/b1) / (1 + (1/b1))
        p2 = (1/b2) / (1 + (1/b2))
        p3 = (1/b3) / (1 + (1/b3))

        return n1, p1, n2, p2, n3, p3, theta1, theta2

def _aic_nb(log_like: float,
            num_param: int) -> float:
    """
    Calculates the Akaike Information Criterion (AIC) score for a
    Negative Binomial Mixture Model

    Parameters:
        log_like (float): log-likelihood value from optimization
        num_params (int): number of parameters in the mixture model
    
    Returns:
        AIC score as a float
    """
    return (-2 * log_like) + 2 * num_param

def _nb_mle_results(counts: list, 
                    ab_name: str,
                    plot: bool = False) -> dict:
    """
    Generates the Negative Binomial results for 1, 2, and 3 component 
    mixture models. Plots the fit of each model on a histogram of the data
    if specified.

    Parameters:
        counts (list): raw counts from protein data
        plot (bool): option to plot each model

    Returns:
        sorted_results (dict): all models and their parameters, sorted by
                               the AIC values in ascending order
        Plots will be printed if plot (bool) is set to True
    """

    results = {}

    for num_components in range(1, 4):
        # Initial parameters for MLE using GMM
        gmm_means, gmm_stdevs, gmm_thetas, \
        *_ = _gmm_init_params(counts, 
                              num_components)
        # GMM
        init_nb, num_comp = _init_params_np(gmm_means, gmm_stdevs, gmm_thetas)

        # Convert GMM parameters to alpha, b 
        conv_init_nb = _convert_np_ab(init_nb, num_comp)

        # Generate bounds for components
        parameter_bounds = _param_bounds(num_comp)

        # Run optimization
        nb_mle = _mle_mix_nb(conv_init_nb, counts, parameter_bounds, num_comp)

        # Convert optimization results to n, p 
        nb_params = _convert_ab_np(nb_mle.x, num_comp)
        
        if num_components == 1:
            n1, p1 = nb_params
            aic = _aic_nb((-1) * nb_mle.fun, 2)
            
            results[num_components] = {
                'num_comp': num_components,
                'model': 'negative binomial (MLE)',
                'aic': aic,
                'nb_n_params': [n1],
                'nb_p_params': [p1],
                'nb_thetas': [1],
            }
        elif num_components == 2:
            n1, p1, n2, p2, theta = nb_params
            aic = _aic_nb((-1) * nb_mle.fun, 5)

            results[num_components] = {
                'num_comp': num_components,
                'model': 'negative binomial (MLE)',
                'aic': aic,
                'nb_n_params': [n1, n2],
                'nb_p_params': [p1, p2],
                'nb_thetas': [theta],
            }
        elif num_components == 3:
            n1, p1, n2, p2, n3, p3, theta1, theta2 = nb_params
            aic = _aic_nb((-1) * nb_mle.fun, 8)

            results[num_components] = {
                'num_comp': num_components,
                'model': 'negative binomial (MLE)',
                'aic': aic,
                'nb_n_params': [n1, n2, n3],
                'nb_p_params': [p1, p2, p3],
                'nb_thetas': [theta1, theta2],
            }
        
    sorted_results = dict(sorted(results.items(), 
                                 key=lambda item: item[1]['aic']))
    if plot:
        _plot_fits(counts=counts,
                  fit_results=sorted_results,
                  ab_name=ab_name)

    return sorted_results