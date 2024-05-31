import seaborn as sns
import plotly.express as px
import umap

from .data_processing import _correlation_ab, PlotUMAPError

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
    corr_df = _correlation_ab(IPD._classified_filt_df, IPD._z_scores_df)
    g = sns.clustermap(corr_df, vmin=-1, vmax=1, cmap='BrBG')
    g.ax_cbar.set_position((1, .2, .03, .4))
 
def plot_UMAP(IPD,
              normalized: bool = False,
              **kwargs):
    """ 
    Plots a UMAP for the non-normalized protein values or normalized protein
    values

    Parameters:
        IPD (ImmunoPhenoData Object): Object containing protein data,
            gene data, and cell types
        normalized (bool): option to plot normalized values
        **kwargs: various arguments to UMAP class constructor, including default values:
            n_neighbors (int): 15
            min_dist (float): 0.1 
            n_components (int): 2
            metric (str): "euclidean"

    Returns:
        UMAP projection of non/normalized protein values with a corresponding
        legend of cell type (if available)
    """
    # Check if existing UMAP is present in class AND UMAP parameters have not changed
    if (IPD._umap_kwargs == kwargs and IPD._raw_umap is not None) and normalized is False:
        # If so, return the stored UMAP
        return IPD._raw_umap
    elif (IPD._umap_kwargs == kwargs and IPD._norm_umap is not None) and normalized is True:
        return IPD._norm_umap
    else:
        # If no UMAP or kwargs are different, generate a new one and store in class
        umap_plot = umap.UMAP(random_state=0, **kwargs)

        # Store new kwargs in class
        IPD._umap_kwargs = kwargs
        
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