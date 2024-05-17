import pytest
import pandas as pd

from src.immunopheno.data_processing import (
    PlotUMAPError
)

from src.immunopheno.plots import (
    plot_antibody_correlation,
    plot_UMAP,
)

def test_antibody_correlation(mocker):
    # Arrange
    mock_IPD = mocker.Mock()
    mock_IPD.classified_filt = pd.DataFrame()
    mock_IPD.z_scores = pd.DataFrame()

    mock_cluster = mocker.patch("src.immunopheno.plots.sns.clustermap")

    # Act
    plot_antibody_correlation(mock_IPD)

    # Assert
    mock_cluster.assert_called_once()

def test_plot_UMAP(mocker):
    # Arrange
    # Mock normalized CITE-seq with labels
    mock_norm_IPD = mocker.MagicMock()
    mock_norm_IPD.normalized_counts = pd.DataFrame(data=[1, 2, 3])
    mock_norm_IPD._cell_labels = pd.DataFrame(data=['1', '2', '3'])
    mock_norm_IPD.labels = pd.DataFrame(data=['1', '2', '3'])
    
    # Mock un-normalized CITE-seq with labels
    mock_norm1_IPD = mocker.MagicMock()
    mock_norm1_IPD.normalized_counts = pd.DataFrame(data=[1, 2, 3])
    mock_norm1_IPD._cell_labels = pd.DataFrame(data=['1', '2', '3'])
    mock_norm1_IPD.labels = pd.DataFrame(data=['1', '2', '3'])

    # Mock Flow Cytometry with no labels
    mock_raw_IPD = mocker.MagicMock()
    mock_raw_IPD.protein = pd.DataFrame(data=[1, 2, 3])
    mock_norm_IPD._cell_labels = None

    # Mock Exception
    mock_error_IPD = mocker.MagicMock()
    mock_error_IPD.normalized_counts = None

    umap = mocker.MagicMock()
    umap.fit_transform(mock_norm_IPD)

    umap1 = mocker.MagicMock()
    umap1.fit_transform(mock_norm1_IPD)

    umap_raw = mocker.MagicMock()
    umap_raw.fit_transform(mock_raw_IPD)

    umap_error = mocker.MagicMock()
    umap_error.fit_transform.side_effect = PlotUMAPError

    mock_UMAP = mocker.patch("src.immunopheno.plots.umap.UMAP",
                             side_effect=[umap, umap1, umap_raw, umap_error])
    
    mock_plot = mocker.patch("src.immunopheno.plots.px.scatter")

    # Act
    plot_UMAP(mock_norm_IPD, True)
    plot_UMAP(mock_norm1_IPD, False)
    plot_UMAP(mock_raw_IPD, True)

    with pytest.raises(PlotUMAPError):
        plot_UMAP(mock_error_IPD, True)
    
    # Assert
    mock_UMAP.assert_called()
    mock_plot.assert_called()