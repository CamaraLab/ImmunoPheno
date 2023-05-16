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
    corr_df = _correlation_ab(IPD.classified_filt, IPD.z_scores)
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
            raise PlotUMAPError("Cannot plot normalized UMAP. "
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