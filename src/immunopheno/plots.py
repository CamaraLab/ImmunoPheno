import seaborn as sns
import plotly.express as px
import logging
import umap

from .data_processing import _correlation_ab, PlotUMAPError, ImmunoPhenoData

def plot_antibody_correlation(IPD: ImmunoPhenoData):
    """Plots a correlation heatmap for each antibody in the data

    Args:
        IPD (ImmunoPhenoData Object): object containing protein data,
            gene data, and cell types
    
    Returns:
        None. Renders a seaborn clustermap for a heatmap of the antibodies.
    """
    # Calculate correlation dataframe
    corr_df = _correlation_ab(IPD._classified_filt_df, IPD._z_scores_df)
    g = sns.clustermap(corr_df, vmin=-1, vmax=1, cmap='BrBG')
    g.ax_cbar.set_position((1, .2, .03, .4))
 
def plot_filter_metrics(IPD: ImmunoPhenoData):
    """Plots a histogram of all D1/D2 distance ratios and entropies returned from run_stvea()

    Args:
        IPD (ImmunoPhenoData Object): object must contain table containing distance ratios and entropies. 
            These are stored in the object after calling run_stvea() from the ImmunoPhenoDB_Connect class. 

    Returns:
        None. Renders two plotly histograms.  
    """
    # Check if distance ratios and entropies exists
    if IPD.distance_ratios is None and IPD.entropies is None:
        raise Exception("Missing distance ratios and entropies. Call run_stvea() or run_dt() first to use plot_filter_metrics()")
    elif IPD.distance_ratios is None and IPD.entropies is not None:
        logging.warning("Unable to find distance_ratios. Only entropies will be plotted")

    ratio = IPD.distance_ratios
    entropies_df = IPD.entropies

    if ratio is not None:
        # Plot a histogram of all ratio values to decide on the ratio_threshold
        ratio_fig = px.histogram(ratio["ratio"], title="D1/D2 Ratios For All Query Cells" + 
                            "<br>" +
                            "<sup>D1: Average distance between nearest neighbor and query cell</sup>" +
                            "<br>" + 
                            "<sup>D2: Average pairwise distance among nearest neighbors</sup>").update_layout(height=500)
        ratio_fig.show()

    if entropies_df is not None:
        entropy_fig = px.histogram(entropies_df["entropy"], title="Entropies For All Query Cells").update_layout(height=500)
        entropy_fig.show()

def plot_UMAP(IPD: ImmunoPhenoData,
              normalized: bool = False,
              force_update: bool = False,
              random_state: int = 42,
              **kwargs):
    """ Plots a UMAP for the non-normalized protein values or normalized protein values

    Args:
        IPD (ImmunoPhenoData Object): object containing protein data,
            gene data, and cell types
        normalized (bool): option to plot normalized values
        force_update (bool): option to compute a new UMAP, replacing stored plots
        random_state (int): seed value for generating the UMAP
        **kwargs: various arguments to UMAP class constructor, including default values:
            n_neighbors (int): 15
            min_dist (float): 0.1 
            n_components (int): 2
            metric (str): "euclidean"

    Returns:
        go.Figure: UMAP projection of non/normalized protein values with a corresponding
        legend of cell type (if available)
    """
    # Check if existing UMAP is present in class AND UMAP parameters have not changed
    if (IPD._umap_kwargs == (random_state, kwargs) and IPD._raw_umap is not None) and normalized is False and force_update is False:
        # If so, return the stored UMAP
        return IPD._raw_umap
    elif (IPD._umap_kwargs == (random_state, kwargs) and IPD._norm_umap is not None) and normalized is True and force_update is False:
        return IPD._norm_umap
    else:
        # If no UMAP or kwargs are different, generate a new one and store in class
        umap_plot = umap.UMAP(random_state=random_state, **kwargs)

        # Store new kwargs in class
        IPD._umap_kwargs = (random_state, kwargs)
        
        if normalized:
            try:
                norm_projections = umap_plot.fit_transform(IPD.normalized_counts)
            except:
                raise PlotUMAPError("Cannot plot normalized UMAP. "
                                "normalize_all_antibodies() must be called first.")
        else:
            raw_projections = umap_plot.fit_transform(IPD.protein)
        
        # Normalized UMAP without cell labels
        if IPD.labels is None and normalized:
            norm_plot = px.scatter(
                norm_projections, x=0, y=1,
            )
            IPD._norm_umap = norm_plot
            return norm_plot
        
        # Un-normalized UMAP without cell labels
        elif IPD._cell_labels is None and not normalized:
            raw_plot = px.scatter(
                raw_projections, x=0, y=1,
            )
            IPD._raw_umap = raw_plot
            return raw_plot

        # Normalized UMAP plot with cell labels
        if IPD.labels is not None and normalized:
            # NOTE: if the provided labels contains more cells than present in normalized_counts
            # Find shared index in the IPD.labels based on cells ONLY in normalized_counts
            common_indices = IPD.normalized_counts.index.intersection(IPD.labels.index)

            # Check the number of columns
            num_columns = IPD.labels.shape[1]
            # Check if there is at least one column and if the second column is not empty
            if num_columns > 1 and not IPD.labels.iloc[:, 1].isnull().all():
                # Use the values from the second column
                norm_types = IPD.labels.iloc[:, 1].loc[common_indices].tolist()

            else:
                # If there is no second column or it is empty, use the values from the first column
                norm_types = IPD.labels.iloc[:, 0].loc[common_indices].tolist()

            norm_plot = px.scatter(
                norm_projections, x=0, y=1,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                color=[str(cell_type) for cell_type in norm_types],
                labels={'color':'cell type'}
            )
            IPD._norm_umap = norm_plot
            return norm_plot
        
        # Not normalized UMAP plot with cell labels
        elif IPD._cell_labels is not None and not normalized:
            # Check the number of columns
            num_columns = IPD._cell_labels.shape[1]
            # Check if there is at least one column and if the second column is not empty
            if num_columns > 1 and not IPD._cell_labels.iloc[:, 1].isnull().all():
                # Use the values from the second column
                raw_types = IPD._cell_labels.iloc[:, 1].tolist()
            else:
                # If there is no second column or it is empty, use the values from the first column
                raw_types = IPD._cell_labels.iloc[:, 0].tolist()
                
            reg_plot = px.scatter(
                raw_projections, x=0, y=1,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                color=[str(cell_type) for cell_type in raw_types],
                labels={'color':'cell type'}
            )
            IPD._raw_umap = reg_plot
            return reg_plot