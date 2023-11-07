import pytest
import pandas as pd
import numpy as np

from src.immunopheno.data_processing import (
    _clean_adt,
    _clean_rna,
    _clean_labels,
    _log_transform,
    _arcsinh_transform,
    _conv_np_mode,
    _find_background_comp,
    _classify_cells,
    _classify_cells_df,
    _filter_classified_df,
    _filter_count_df,
    _filter_cell_labels,
    _z_scores,
    _z_scores_df,
    _bg_z_scores_df,
    _z_avg_umi_sum,
    _z_avg_umi_sum_by_type,
    _linear_reg,
    _linear_reg_by_type,
    _get_cell_type,
    _bg_mean_std,
    _correlation_ab,
    _normalize_antibody,
    _normalize_antibodies_df,
    ImmunoPhenoData,
    LoadMatrixError,
    InvalidModelError,
    ExtraArgumentsError,
    TransformTypeError,
    TransformScaleError,
    PlotAntibodyFitError,
    PlotPercentileError,
    AntibodyLookupError,
    EmptyAntibodyFitsError,
    IncompleteAntibodyFitsError,
    BoundsThresholdError,
    BackgroundZScoreError
)

@pytest.fixture
def gmm_fit():
    return {3: {'num_comp': 3,
                'model': 'gaussian (EM)',
                'aic': 8.84,
                'gmm_means': [4.99, 2.00, 3.99],
                'gmm_stdevs': [0.001, 0.81, 0.001],
                'gmm_thetas': [0.19, 0.60, 0.19],
                'gmm_log_like': 3.57}}

@pytest.fixture
def nb_fit():
    return {3: {'num_comp': 3,
                'model': 'negative binomial (MLE)',
                'aic': 32.86,
                'nb_n_params': [99999.99, 899999.99, 399999.99],
                'nb_p_params': [0.70, 0.99, 0.93],
                'nb_thetas': [0.25, 0.48]}}

def test_clean_adt():
    # Arrange
    count_matrix = pd.DataFrame(index=['Antibody'], columns=['Cell'])
    expected = count_matrix.T

    # Act
    output = _clean_adt(count_matrix)

    # Assert
    assert output.index == expected.index
    assert output.columns == expected.columns

def test_clean_rna():
    # Arrange
    gene_matrix = pd.DataFrame(index=['Gene'], columns=['Cell'])
    expected = gene_matrix.T

    # Act
    output = _clean_rna(gene_matrix)

    # Assert
    assert output.index == expected.index
    assert output.columns == expected.columns

def test_clean_labels():
    # Arrange
    label_matrix = pd.DataFrame(data=[['cell1', 'NK'], ['cell2', 'B']])
    expected_index = label_matrix.iloc[:, 0]

    # Act
    output = _clean_labels(label_matrix)

    # Assert
    assert list(output.index) == list(expected_index)

def test_log_transform():
    # Arrange
    counts = [10, 20, 30]
    expected = [0.0, 2.39, 3.04]

    # Act
    output = list(_log_transform(counts))

    # Assert
    assert output == pytest.approx(expected, 0.2)

def test_arcsinh_transform():
    # Arrange
    counts = [10, 20, 30]
    expected = [0.0, 2.99, 3.68]

    # Act
    output = list(_arcsinh_transform(counts))

    # Assert
    assert output == pytest.approx(expected, 0.2)

@pytest.mark.parametrize(
    "n, p, expected",
    [
        (100, 0.3, 231.0),
        (0.1, 0.3, 0)
    ]
)
def test_conv_np_mode(n, p, expected):
    assert _conv_np_mode(n, p) == expected

@pytest.mark.parametrize(
    "fit_results, expected",
    [
        ({3: {'num_comp': 3,
                'model': 'gaussian (EM)',
                'aic': 8.84,
                'gmm_means': [4.99, 2.00, 3.99],
                'gmm_stdevs': [0.001, 0.81, 0.001],
                'gmm_thetas': [0.19, 0.60, 0.19],
                'gmm_log_like': 3.57}}, 1),
        ({2: {'num_comp': 2,
                'model': 'negative binomial (MLE)',
                'aic': 30.659,
                'nb_n_params': [225010.00, 899999.99],
                'nb_p_params': [0.85, 0.99],
                'nb_thetas': [0.66]}}, 1)
    ]
)
def test_find_background_comp(fit_results, expected):
    assert _find_background_comp(fit_results) == expected

@pytest.mark.parametrize(
    "fit_results, data_vector, bg_comp, expected",
    [
        ({3: {'num_comp': 3,
                'model': 'gaussian (EM)',
                'aic': 8.84,
                'gmm_means': [4.99, 2.00, 3.99],
                'gmm_stdevs': [0.001, 0.81, 0.001],
                'gmm_thetas': [0.19, 0.60, 0.19],
                'gmm_log_like': 3.57}}, [153, 3, 63], 1, [0, 0, 0]),
        ({2: {'num_comp': 2,
                'model': 'gaussian (EM)',
                'aic': 158.55,
                'gmm_means': [2.52, 40.30],
                'gmm_stdevs': [2.17, 13.00],
                'gmm_thetas': [0.73, 0.26],
                'gmm_log_like': -74.27}}, [75, 23, 12, 55], 0, [1, 1, 1, 1]),
        ({3: {'num_comp': 3,
                'model': 'negative binomial (MLE)',
                'aic': 32.86,
                'nb_n_params': [99999.99, 899999.99, 399999.99],
                'nb_p_params': [0.70, 0.99, 0.93],
                'nb_thetas': [0.25, 0.48]}}, [25, 77, 2], 1, [0, 0, 0]),
        ({2: {'num_comp': 2,
                'model': 'negative binomial (MLE)',
                'aic': 30.659,
                'nb_n_params': [225010.00, 899999.99],
                'nb_p_params': [0.85, 0.99],
                'nb_thetas': [0.66]}}, [22, 244, 4], 1, [0, 0, 0])
    ]
)
def test_classify_cells(fit_results, data_vector, bg_comp, expected):
    assert _classify_cells(fit_results, data_vector, bg_comp) == expected


def test_classify_cells_df(mocker, gmm_fit):
    # Arrange
    fits = [gmm_fit]
    
    counts_df = pd.DataFrame({'ab1': [153, 3, 63]})

    mock_background = mocker.patch("src.immunopheno.data_processing._find_background_comp",
                                   return_value=1)

    mock_classify_cells = mocker.patch("src.immunopheno.data_processing._classify_cells",
                                       return_value=pd.DataFrame([0, 0, 0]))
    
    expected = pd.DataFrame({'ab1': [0, 0, 0]})
    
    # Act
    output = _classify_cells_df(fits, counts_df)

    # Assert
    pd.testing.assert_frame_equal(output, expected)

def test_filter_classified_df():
    # Arrange
    classified = pd.DataFrame(data=[[1, 1, 0, 1, 1, 0, 1],
                                    [0, 1, 0, 0, 0, 0, 0]])
    
    expected = pd.DataFrame(data=[[1, 1, 0, 1, 1, 0, 1]])

    # Act
    output = _filter_classified_df(classified)

    # Assert
    pd.testing.assert_frame_equal(output, expected)

def test_filter_count_df():
    # Arrange
    classified = pd.DataFrame(data=[[1, 1, 0, 1, 1, 0, 1]])

    counts = pd.DataFrame(data=[[32, 5, 6, 123, 78, 23, 0], 
                                [85, 23, 9, 64, 13, 65, 2]])
    
    expected = pd.DataFrame(data=[[32, 5, 6, 123, 78, 23, 0]])

    # Act
    output = _filter_count_df(classified, counts)

    # Assert
    pd.testing.assert_frame_equal(output, expected)
    
def test_filter_cell_labels():
    # Arrange
    classified = pd.DataFrame(data=[[1, 1, 0, 1, 1, 0, 1]])
    
    labels = pd.DataFrame(data=[['NK'], ['T'], ['B']])

    expected = pd.DataFrame(data=[['NK']])

    # Act
    output = _filter_cell_labels(classified, labels)

    # Assert
    pd.testing.assert_frame_equal(output, expected)

def test_z_scores(gmm_fit):
    # Arrange
    counts = [1, 0, 3, 8, 6]

    expected = [-0.86, -1.19, -0.19, 1.46, 0.79]

    # Act
    output = _z_scores(gmm_fit, counts)

    # Assert
    assert output == pytest.approx(expected, 0.2)

def test_z_scores_df(mocker, gmm_fit):
    # Arrange
    fits = [gmm_fit]
    counts_df = pd.DataFrame({'ab1': [1, 0, 3, 8, 6]})

    mock_z_scores = mocker.patch("src.immunopheno.data_processing._z_scores",
                                 return_value=[-0.86, -1.19, -0.19, 1.46, 0.79])
    
    expected = pd.DataFrame({'ab1': [-0.86, -1.19, -0.19, 1.46, 0.79]})

    # Act
    output = _z_scores_df(fits, counts_df)

    # Assert
    pd.testing.assert_frame_equal(output, expected)

def test_bg_z_scores_df():
    # Arrange
    classified = pd.DataFrame({'ab1': [1, 1, 0, 1, 0]})
    z_scores = pd.DataFrame({'ab1': [-0.86, -1.19, -0.19, 1.46, 0.79]})
    expected = pd.DataFrame({'ab1': [np.nan, np.nan, -0.19, np.nan, 0.79]})

    # Act
    output = _bg_z_scores_df(classified, z_scores)

    # Assert
    pd.testing.assert_frame_equal(output, expected)

def test_z_avg_umi_sum():
    # Arrange
    bg_z_scores = pd.DataFrame({'ab1': [np.nan, np.nan, -0.19, np.nan, 0.79]})
    rna_counts = pd.DataFrame([[343, 5, 23, 154], 
                               [235, 35, 23, 1],
                               [53, 8, 35, 1],
                               [23, 6, 11, 9],
                               [3, 99, 23, 1]])
    
    expected_data = {'z_score_avg': [-0.19, 0.79],
                     'total_umi': [97, 126]}
    expected_df = pd.DataFrame(expected_data, index=[2, 4])

    # Act
    output = _z_avg_umi_sum(bg_z_scores, rna_counts)

    # Assert
    pd.testing.assert_frame_equal(output, expected_df)

def test_z_avg_umi_sum_by_type():
    # Arrange
    bg_z_scores = pd.DataFrame({'ab1': [np.nan, np.nan, -0.19, np.nan, 0.79]})
    rna_counts = pd.DataFrame([[343, 5, 23, 154], 
                               [235, 35, 23, 1],
                               [53, 8, 35, 1],
                               [23, 6, 11, 9],
                               [3, 99, 23, 1]])
    labels = pd.DataFrame(data=[['NK'], ['T'], ['B'], ['T'], ['NK']])

    row1 = {'z_score_avg': -0.19, 'total_umi': 97, 'type': 'B'}
    df1 = pd.DataFrame(data=row1, 
                       index=[2], 
                       columns=['z_score_avg', 'total_umi', 'type'])
    
    row2= {'z_score_avg': 0.79, 'total_umi': 126, 'type': 'NK'}
    df2 = pd.DataFrame(data=row2,
                       index=[4],
                       columns=['z_score_avg', 'total_umi', 'type'])
    
    df3 = pd.DataFrame(columns=['z_score_avg', 'total_umi', 'type'])

    expected = {'B': df1,
                'NK': df2,
                'T': df3}

    # Act
    output = _z_avg_umi_sum_by_type(bg_z_scores, rna_counts, labels)

    # Assert
    assert set(output.keys()) == set(expected.keys())
    pd.testing.assert_frame_equal(output['B'], expected['B'])
    pd.testing.assert_frame_equal(output['NK'], expected['NK'])
    pd.testing.assert_frame_equal(output['T'], expected['T'],
                                  check_dtype=False,
                                  check_index_type=False)
    
def test_linear_reg():
    # Arrange
    z_umi = pd.DataFrame(data={'z_score_avg': [-0.19, 0.79, 0.4, 0.23],
                               'total_umi': [97, 126, 77, 123]})

    expected = pd.DataFrame(data={'z_score_avg': [-0.19, 0.79, 0.4, 0.23],
                                  'total_umi': [97, 126, 77, 123],
                                  'predicted': [0.27, 0.41, 0.14, 0.39],
                                  'residual': [-0.46, 0.37, 0.25, -0.16],
                                  'p_val': [0.69, 0.69, 0.69, 0.69]})
    # Act
    output = _linear_reg(z_umi)

    # Assert
    pd.testing.assert_frame_equal(output, expected,
                                  atol=0.2)
    
def test_linear_reg_by_type():
    # Arrange
    z_umi_1 = {'z_score_avg': [-0.19, 0.79, -0.92, -2.3],
               'total_umi': [97, 11, 45, 63],
               'type': 'B'}
    df1 = pd.DataFrame(data=z_umi_1, columns=['z_score_avg', 'total_umi', 'type'])
    
    z_umi_2 = {'z_score_avg': [-0.56, 0.19, -0.42, -1.3],
               'total_umi': [33, 91, 41, 83],
               'type': 'NK'}
    df2 = pd.DataFrame(data=z_umi_2, columns=['z_score_avg', 'total_umi', 'type'])
    
    df3 = pd.DataFrame(columns=['z_score_avg', 'total_umi', 'type'])

    df_dict = {'B': df1,
               'NK': df2,
               'T': df3}
    
    lin_reg_B = pd.DataFrame(data={'z_score_avg': [-0.19, 0.79, -0.92, -2.30],
                                   'total_umi': [97, 11, 45, 63],
                                   'type': ['B', 'B', 'B', 'B'],
                                   'predicted': [-1.36, 0.46, -0.71, -1.00],
                                   'residual': [1.17, 0.32, -0.20, -1.29],
                                   'p_val': [0.39, 0.39, 0.39, 0.39]})

    lin_reg_NK = pd.DataFrame(data={'z_score_avg': [-0.56, 0.19, -0.42, -1.30],
                                    'total_umi': [33, 91, 41, 83],
                                    'type': ['NK', 'NK', 'NK', 'NK'],
                                    'predicted': [-0.54, -0.50, -0.53, -0.50],
                                    'residual': [-0.01, 0.69, 0.11, -0.79],
                                    'p_val': [0.96, 0.96, 0.96, 0.96]})
    
    expected = {'B': lin_reg_B,
                'NK': lin_reg_NK}

    # Act
    output = _linear_reg_by_type(df_dict)

    # Assert
    assert set(output.keys()) == set(expected.keys())
    pd.testing.assert_frame_equal(output['B'], expected['B'],
                                  atol=0.35)
    pd.testing.assert_frame_equal(output['NK'], expected['NK'],
                                  atol=0.35)

def test_get_cell_type():
    # Arrange
    cell_labels = pd.DataFrame(data=[['NK'], ['T'], ['B']], 
                               index=['cell1', 'cell2', 'cell3'])
    cell_name = 'cell1'
    expected = 'NK'

    # Act
    output = _get_cell_type(cell_name, cell_labels)

    # Assert
    assert output == expected

@pytest.mark.parametrize(
    "fit_results, expected",
    [
        ({3: {'num_comp': 3,
                'model': 'gaussian (EM)',
                'aic': 8.84,
                'gmm_means': [4.99, 2.00, 3.99],
                'gmm_stdevs': [0.001, 0.81, 0.001],
                'gmm_thetas': [0.19, 0.60, 0.19],
                'gmm_log_like': 3.57}}, ((2.0, 0.81))),
        ({2: {'num_comp': 2,
                'model': 'negative binomial (MLE)',
                'aic': 30.659,
                'nb_n_params': [225010.00, 899999.99],
                'nb_p_params': [0.85, 0.99],
                'nb_thetas': [0.66]}}, (9090.90, 95.82))
    ]
)
def test_bg_mean_std(fit_results, expected):
    assert _bg_mean_std(fit_results) == pytest.approx(expected, 0.2)

def test_correlation_ab():
    # Arrange
    ab_class = {'ab1': [1, 0, 0, 0, 0], 
                'ab2': [0, 0, 0, 0, 1],
                'ab3': [0, 1, 0, 0, 0],
                'ab4': [1, 0, 0, 0, 0]}
    ab_df = pd.DataFrame(data=ab_class, 
                         index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    z_scores = {'ab1': [0.1, -2.3, -1.3, 0.4, -0.8],
                'ab2': [1.76, 1.3, 0.6, -0.3, -2.1],
                'ab3': [-1.5, 2.1, 1.3, 1.7, -0.1],
                'ab4': [2.1, 0.2, -0.3, 0.9, 1.9]}
    z_scores_df = pd.DataFrame(data=z_scores,
                               index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])

    expected_corr = {'ab1': [1.0, -0.99, 0.43, 0.45], 
                     'ab2': [-0.99, 1.0, -0.94, -0.63],
                     'ab3': [0.43, -0.94, 1.0, -0.70],
                     'ab4': [0.45, -0.63, -0.70, 1.0]}
    
    expected_df = pd.DataFrame(data=expected_corr,
                               index=['ab1', 'ab2', 'ab3', 'ab4'])

    # Act
    output = _correlation_ab(ab_df, z_scores_df)

    # Assert
    pd.testing.assert_frame_equal(output, expected_df,
                                  atol=0.25)
    
def test_normalize_antibody(gmm_fit):
    # Arrange
    counts = pd.Series(data=[243, 334, 829, 546, 617],
                       index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    ab_class = {'ab1': [1, 0, 1, 1, 1], 
                'ab2': [1, 0, 1, 1, 1],
                'ab3': [0, 1, 1, 1, 0],
                'ab4': [1, 0, 1, 1, 0]}
    ab_df = pd.DataFrame(data=ab_class, 
                         index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    labels = pd.DataFrame(data=['NK', 'B', 'NK', 'B', 'NK'], 
                          index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    general_lin_reg = pd.DataFrame(data={'z_score_avg': [-1.9, 2.79, -2.92, -1.30, -0.32],
                                         'total_umi': [523, 413, 214, 23, 90],
                                         'predicted': [-0.16, 2.46, -1.71, 0.52, -1.01],
                                         'residual': [3.17, 0.32, -1.20, 0.99, -4.28],
                                         'p_val': [0.01, 0.01, 0.01, 0.01, 0.01]},
                                   index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])

    lin_reg_B = pd.DataFrame(data={'z_score_avg': [-0.19, 0.79, -0.92, -2.30, -2.32],
                                   'total_umi': [97, 11, 45, 63, 62],
                                   'type': ['B', 'B', 'B', 'B', 'B'],
                                   'predicted': [-1.36, 0.46, -0.71, -1.00, -1],
                                   'residual': [1.17, 0.32, -0.20, -1.29, -1.28],
                                   'p_val': [0.01, 0.01, 0.01, 0.01, 0.01]},
                             index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])

    lin_reg_NK = pd.DataFrame(data={'z_score_avg': [-0.56, 0.19, -0.42, -1.30, -1.20],
                                    'total_umi': [33, 91, 41, 83, 81],
                                    'type': ['NK', 'NK', 'NK', 'NK', 'NK'],
                                    'predicted': [-213.54, -0.50, -0.53, -0.50, -0.52],
                                    'residual': [-0.01, 0.69, 0.11, -0.79, -0.71],
                                    'p_val': [0.07, 0.07, 0.07, 0.07, 0.07]},
                             index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    lin_reg_dict = {'B': lin_reg_B,
                    'NK': lin_reg_NK}
    
    expected_by_type = [-1.07, -10, 1.31, -0.32, 0.08]
    expected_without_type = [-10, -0.94, 1.04, -0.09, -10]
    expected_flow = [-1.30, -10, 1.11, -0.05, 0.24]
    
    # Act
    output_by_type = _normalize_antibody(fit_results=gmm_fit,
                                         data_vector=counts,
                                         ab_name='ab1',
                                         p_threshold=0.1,
                                         background_cell_z_score=-10,
                                         classified_filt_df=ab_df,
                                         cell_labels_filt_df=labels,
                                         lin_reg_dict=lin_reg_dict)
    
    output_without_type = _normalize_antibody(fit_results=gmm_fit,
                                              data_vector=counts,
                                              ab_name='ab3',
                                              p_threshold=0.1,
                                              background_cell_z_score=-10,
                                              classified_filt_df=ab_df,
                                              lin_reg=general_lin_reg)
    
    output_flow = _normalize_antibody(fit_results=gmm_fit,
                                      data_vector=counts,
                                      ab_name='ab2',
                                      p_threshold=0.1,
                                      background_cell_z_score=-10,
                                      classified_filt_df=ab_df)

    # Assert
    assert output_by_type == pytest.approx(expected_by_type, 0.2)
    assert output_without_type == pytest.approx(expected_without_type, 0.2)
    assert output_flow == pytest.approx(expected_flow, 0.2)

def test_normalize_antibodies_df(mocker, gmm_fit):
    # Arrange
    counts = pd.DataFrame(data=[243, 334, 829, 546, 617],
                          columns=['ab2'],
                          index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    ab_class = {'ab1': [1, 0, 1, 1, 1], 
                'ab2': [1, 0, 1, 1, 1],
                'ab3': [0, 1, 1, 1, 0],
                'ab4': [1, 0, 1, 1, 0]}
    
    ab_df = pd.DataFrame(data=ab_class, 
                         index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    mock_norm = mocker.patch("src.immunopheno.data_processing._normalize_antibody",
                             return_value=[-1.30, -10, 1.11, -0.05, 0.24])

    expected = pd.DataFrame(data=[-1.30, -10, 1.11, -0.05, 0.24],
                            columns=['ab2'],
                            index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    # Act
    output = _normalize_antibodies_df(protein_cleaned_filt_df=counts,
                                      fit_all_results=[gmm_fit],
                                      classified_filt_df=ab_df)

    # Assert
    pd.testing.assert_frame_equal(output, expected,
                                  atol=0.2)

@pytest.fixture()
def raw_cite_protein():
    raw_counts = pd.DataFrame(data=[[243, 334, 829, 546, 617]],
                              columns=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'],
                              index=['ab1'])
    return raw_counts

@pytest.fixture()
def raw_cite_rna():
    raw_genes = pd.DataFrame([[343, 235, 53, 23, 3],
                              [5, 35, 8, 6, 99],
                              [23, 23, 35, 11, 23],
                              [154, 1, 1, 9, 1]],
                              index=['gene0', 'gene1', 'gene2', 'gene3'],
                              columns=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    return raw_genes

@pytest.fixture()
def raw_labels():
    labels = pd.DataFrame(data=[['cell0', 'NK'], 
                                ['cell1', 'B'], 
                                ['cell2', 'NK'], 
                                ['cell3', 'B'], 
                                ['cell4', 'NK']])
    return labels

@pytest.fixture()
def raw_flow_protein():
    raw_flow_counts =  pd.DataFrame(data=[243, 334, 829, 546, 617],
                                    columns=['ab1'],
                                    index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    return raw_flow_counts

@pytest.mark.parametrize(
    'protein_matrix, gene_matrix, cell_labels',
    [
        (None, None, None),
        (pd.DataFrame(),None, pd.DataFrame()),
    ]
)
def test_load_ImmunoPhenoData(protein_matrix, 
                              gene_matrix, 
                              cell_labels):
    
    with pytest.raises(LoadMatrixError):
        ipd1 = ImmunoPhenoData(protein_matrix, gene_matrix, cell_labels)
    
    with pytest.raises(LoadMatrixError):
        ipd2 = ImmunoPhenoData(protein_matrix, gene_matrix, cell_labels)


def test_clean_ImmunoPhenoData(mocker,
                               raw_cite_protein,
                               raw_cite_rna,
                               raw_flow_protein,
                               raw_labels):
    # Arrange
    clean_labels = pd.DataFrame(data=[['NK'], ['B'], ['NK'], ['B'], ['NK']],
                                index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])

    mock_adt = mocker.patch("src.immunopheno.data_processing._clean_adt",
                            return_value=raw_cite_protein.T)
    
    mock_rna = mocker.patch("src.immunopheno.data_processing._clean_rna",
                            return_value=raw_cite_rna.T)
    
    mock_labels = mocker.patch("src.immunopheno.data_processing._clean_labels",
                               return_value=clean_labels)
    
    # Act
    ipd_cite = ImmunoPhenoData(protein_matrix=raw_cite_protein,
                               gene_matrix=raw_cite_rna,
                               cell_labels=None)
    
    ipd_cite_labels = ImmunoPhenoData(protein_matrix=raw_cite_protein,
                                      gene_matrix=raw_cite_rna,
                                      cell_labels=raw_labels)
    
    ipd_flow = ImmunoPhenoData(protein_matrix=raw_flow_protein,
                               gene_matrix=None,
                               cell_labels=None)
    
    # Assert
    pd.testing.assert_frame_equal(ipd_cite._protein_matrix, raw_cite_protein.T)
    pd.testing.assert_frame_equal(ipd_cite._gene_matrix, raw_cite_rna.T)
    assert ipd_cite._cell_labels == None

    pd.testing.assert_frame_equal(ipd_cite_labels._protein_matrix, raw_cite_protein.T)
    pd.testing.assert_frame_equal(ipd_cite_labels._gene_matrix, raw_cite_rna.T)
    pd.testing.assert_frame_equal(ipd_cite_labels._cell_labels, clean_labels)

    pd.testing.assert_frame_equal(ipd_flow._protein_matrix, raw_flow_protein)
    assert ipd_flow._gene_matrix == None
    assert ipd_flow._cell_labels == None

def test_property_ImmunoPhenoData():
    # Arrange
    ipd = ImmunoPhenoData(pd.DataFrame())
    ipd._classified_filt_df = pd.DataFrame()
    ipd._z_scores_df = pd.DataFrame()
    ipd._normalized_counts_df = pd.DataFrame()
    ipd._cell_labels_filt_df= pd.DataFrame()
    ipd._cell_labels = pd.DataFrame()

    # Act
    output_classified = ipd.classified_filt
    output_z_scores = ipd.z_scores
    output_norm_counts = ipd.normalized_counts
    output_protein = ipd.protein
    output_norm_labels = ipd.norm_cell_labels
    output_raw_labels = ipd.raw_cell_labels

    expected = pd.DataFrame()

    # Assert
    pd.testing.assert_frame_equal(output_classified, expected)
    pd.testing.assert_frame_equal(output_z_scores, expected)
    pd.testing.assert_frame_equal(output_norm_counts, expected)
    pd.testing.assert_frame_equal(output_protein, expected)
    pd.testing.assert_frame_equal(output_norm_labels, expected)
    pd.testing.assert_frame_equal(output_raw_labels, expected)

def test_fit_antibody(mocker,
                      raw_cite_protein,
                      raw_cite_rna,
                      raw_labels,
                      gmm_fit,
                      nb_fit):
    # Arrange
    ipd = ImmunoPhenoData(raw_cite_protein, raw_cite_rna, raw_labels)

    counts = [243, 334, 829, 546, 617]

    clean_labels = pd.DataFrame(data=[['NK'], ['B'], ['NK'], ['B'], ['NK']],
                                index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])

    log_expected = [0.0, 4.52, 6.37, 5.71, 5.92]
    
    mock_adt = mocker.patch("src.immunopheno.data_processing._clean_adt",
                            return_value=raw_cite_protein.T)
    
    mock_rna = mocker.patch("src.immunopheno.data_processing._clean_rna",
                            return_value=raw_cite_rna.T)
    
    mock_labels = mocker.patch("src.immunopheno.data_processing._clean_labels",
                               return_value=clean_labels)
    
    mock_log = mocker.patch("src.immunopheno.data_processing._log_transform",
                            return_value=log_expected)
    
    mock_nb = mocker.patch("src.immunopheno.data_processing._nb_mle_results",
                           return_value=nb_fit)
    
    ipd_log_gauss = ipd.fit_antibody(input="ab1", 
                                     model="nb", 
                                     transform_type="log")

    ipd_flow = ImmunoPhenoData(raw_cite_protein.T)
    
    arcsinh_expected = [0.0, 5.20, 7.06, 6.40, 6.61]
    
    mock_arcsinh = mocker.patch("src.immunopheno.data_processing._arcsinh_transform",
                                return_value=arcsinh_expected)
    
    mock_gmm = mocker.patch("src.immunopheno.data_processing._gmm_results",
                                return_value=gmm_fit)
    
    ipd_arcsinh_nb = ipd_flow.fit_antibody(input=[243, 334, 829, 546, 617],
                                           model="gaussian",
                                           transform_type="arcsinh")

    # Assert
    assert ipd_log_gauss == nb_fit
    assert ipd_arcsinh_nb == gmm_fit

    with pytest.raises(InvalidModelError):
        ipd.fit_antibody(input=counts, model="error")
    
    with pytest.raises(ExtraArgumentsError):
        ipd.fit_antibody(input=counts, model="nb", kwargs=True)
    
    with pytest.raises(TransformTypeError):
        ipd.fit_antibody(input=counts, transform_scale=2, transform_type=None)

    with pytest.raises(TransformScaleError):
        ipd.fit_antibody(input=counts, transform_type="nb", transform_scale=1.05)
    
    with pytest.raises(PlotAntibodyFitError):
        ipd.fit_antibody(input=counts, model="nb", plot="error")
    
    with pytest.raises(PlotPercentileError):
        ipd.fit_antibody(input=counts, plot_percentile="23")
    
    with pytest.raises(AntibodyLookupError):
        ipd.fit_antibody(input="ab_error")
    
    with pytest.raises(TransformTypeError):
        ipd.fit_antibody(input=counts, transform_type="error_transform")

def test_fit_all_antibodies(mocker, raw_cite_protein, gmm_fit):
    # Arrange (flow data)
    mock_fit_antibody = mocker.patch("src.immunopheno.data_processing.ImmunoPhenoData.fit_antibody",
                                     return_value=gmm_fit)

    # Act
    ipd = ImmunoPhenoData(raw_cite_protein.T)
    ipd._all_fits_dict = {'ab1': None}

    output = ipd.fit_all_antibodies()

    expected = [gmm_fit]

    # Assert
    assert output == expected

def test_normalize_all_antibodies(mocker,
                                  raw_cite_protein,
                                  raw_cite_rna,
                                  raw_labels):
    
    ipd = ImmunoPhenoData(raw_cite_protein)
    with pytest.raises(EmptyAntibodyFitsError):
        ipd.normalize_all_antibodies()
    
    ipd._all_fits = ['fit']
    ipd._all_fits_dict = {'ab1': None}
    with pytest.raises(IncompleteAntibodyFitsError):
        ipd.normalize_all_antibodies()
    
    ipd._all_fits_dict = {'ab1': 'fit'}
    with pytest.raises(BoundsThresholdError):
        ipd.normalize_all_antibodies(p_threshold=100, 
                                     sig_expr_threshold=100, 
                                     bg_expr_threshold=100)
        
    with pytest.raises(BackgroundZScoreError):
        ipd.normalize_all_antibodies(p_threshold=0.1, 
                                     sig_expr_threshold=0.1, 
                                     bg_expr_threshold=0.1,
                                     bg_cell_z_score=100)
    
    ab_class = {'ab1': [1, 0, 1, 1, 1], 
                'ab2': [1, 0, 1, 1, 1],
                'ab3': [0, 1, 1, 1, 0],
                'ab4': [1, 0, 1, 1, 0]}
    
    ab_df = pd.DataFrame(data=ab_class, 
                         index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    clean_labels = pd.DataFrame(data=[['NK'], ['B'], ['NK'], ['B'], ['NK']],
                                index=['cell0', 'cell1', 'cell2', 'cell3', 'cell4'])
    
    expected = pd.DataFrame([1, 2, 3, 4, 5])
    
    ipd_cite_labels = ImmunoPhenoData(raw_cite_protein,
                                      raw_cite_rna,
                                      raw_labels)
    
    ipd_cite_labels._all_fits = ['fit']
    ipd_cite_labels._all_fits_dict = {'ab1': 'fit'}
    
    mock_classify_cells = mocker.patch("src.immunopheno.data_processing._classify_cells_df",
                                       return_value=ab_df)
    
    mock_filt_classify = mocker.patch("src.immunopheno.data_processing._filter_classified_df",
                                      return_value=ab_df)
    
    mock_filter_count = mocker.patch("src.immunopheno.data_processing._filter_count_df",
                                     return_value=raw_cite_protein.T)
    
    mock_filter_labels = mocker.patch("src.immunopheno.data_processing._filter_cell_labels",
                                      return_value=clean_labels) 
    
    mock_z_scores = mocker.patch("src.immunopheno.data_processing._z_scores_df",
                                 return_value=pd.DataFrame())
    
    mock_filt_z_scores = mocker.patch("src.immunopheno.data_processing._bg_z_scores_df",
                                      return_value=pd.DataFrame())
    
    mock_z_avg_umi = mocker.patch("src.immunopheno.data_processing._z_avg_umi_sum",
                                  return_value=pd.DataFrame())
    
    mock_z_avg_umi_type = mocker.patch("src.immunopheno.data_processing._z_avg_umi_sum_by_type",
                                       return_value=pd.DataFrame())
    
    mock_lin_reg = mocker.patch("src.immunopheno.data_processing._linear_reg",
                                return_value=pd.DataFrame())
    
    mock_lin_reg_type = mocker.patch("src.immunopheno.data_processing._linear_reg_by_type",
                                     return_value=pd.DataFrame())
    
    mock_normalize_antibodies_df = mocker.patch("src.immunopheno.data_processing._normalize_antibodies_df",
                                                return_value=pd.DataFrame([1, 2, 3, 4, 5]))

    output_ipd_cite_labels = ipd_cite_labels.normalize_all_antibodies()

    ipd_cite_no_labels = ImmunoPhenoData(raw_cite_protein,
                                         raw_cite_rna)
    ipd_cite_no_labels._all_fits = ['fit']
    ipd_cite_no_labels._all_fits_dict = {'ab1': 'fit'}
    output_ipd_cite_no_labels = ipd_cite_no_labels.normalize_all_antibodies()

    ipd_flow = ImmunoPhenoData(raw_cite_protein)
    ipd_flow._all_fits = ['fit']
    ipd_flow._all_fits_dict = {'ab1': 'fit'}
    output_ipd_flow = ipd_flow.normalize_all_antibodies()

    pd.testing.assert_frame_equal(output_ipd_cite_labels, expected)
    pd.testing.assert_frame_equal(output_ipd_cite_no_labels, expected)
    pd.testing.assert_frame_equal(output_ipd_flow, expected)