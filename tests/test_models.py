import pytest
import numpy as np
import pandas as pd

from src.immunopheno.models import (
    plot_fits,
    plot_all_fits,
    _gmm_init_params,
    _gmm_results,
    _convert_gmm_np, 
    _init_params_np,
    _param_bounds,
    _convert_np_ab,
    _log_like_nb,
    _theta_constr_mle_2,
    _theta_constr_mle_3,
    _mle_mix_nb,
    _convert_ab_np,
    _aic_nb,
    _nb_mle_results
)

@pytest.fixture
def example_fit_results():
    return {3: {'num_comp': 3,
                'model': 'gaussian (EM)',
                'aic': 8.84,
                'gmm_means': [4.99, 2.00, 3.99],
                'gmm_stdevs': [0.001, 0.81, 0.001],
                'gmm_thetas': [0.19, 0.60, 0.19],
                'gmm_log_like': 3.57},
            1: {'num_comp': 1,
                'model': 'gaussian (EM)',
                'aic': 21.65,
                'gmm_means': [2.99],
                'gmm_stdevs': [1.41],
                'gmm_thetas': [1.0],
                'gmm_log_like': -8.82},
            2: {'num_comp': 2,
                'model': 'negative binomial (MLE)',
                'aic': 30.659,
                'nb_n_params': [225010.00, 899999.99],
                'nb_p_params': [0.85, 0.99],
                'nb_thetas': [0.66]}}

def test_plot_fits(mocker, example_fit_results):
    # Arrange
    counts = [1, 2, 3, 4, 5]
    mock_plots = mocker.patch("src.immunopheno.models.plt.show")
    
    # Act
    plot_fits(counts, example_fit_results)

    # Assert
    mock_plots.assert_called_once()

def test_plot_all_fits(mocker, example_fit_results):
    # Arrange
    mock_IPD = mocker.Mock()
    mock_IPD.protein = pd.DataFrame({'ab1': [153, 235, 4, 2]})
    mock_IPD._all_fits = [example_fit_results]

    mock_all_plots = mocker.patch("src.immunopheno.models.plot_fits")

    # Act
    plot_all_fits(mock_IPD)

    # Assert
    mock_all_plots.assert_called()

def test_gmm_init_params():
    # Arrange
    counts = [1, 2, 3, 4, 5]
    n = 2

    expected_means = [4.50, 2.07]
    expected_stdevs = [0.53, 0.90]
    expected_thetas = [0.37, 0.62]
    expected_log = -8.33
    expected_aic = 26.66

    # Act
    output_means, output_stdevs, output_thetas, output_log, output_aic = _gmm_init_params(counts, n)

    # Assert
    assert output_means == pytest.approx(expected_means, 0.4)
    assert output_stdevs == pytest.approx(expected_stdevs, 0.4)
    assert output_thetas == pytest.approx(expected_thetas, 0.4)
    assert output_log == pytest.approx(expected_log, 0.4)
    assert output_aic == pytest.approx(expected_aic, 0.4)

def test_gmm_results(example_fit_results):
    # Arrange
    counts = [1, 2, 3, 4, 5]
    
    # Act
    output = _gmm_results(counts)

    # Assert
    assert example_fit_results[3]['aic'] == pytest.approx(output[3]['aic'], 0.2)

@pytest.mark.parametrize(
    "mean, stdev, expected",
    [
        (8.0, 4.0, (8.0, 0.5)),
        (12.0, 2.0, (14400000.00, 0.99))
    ]
)
def test_convert_gmm_np(mean, stdev, expected):
    assert _convert_gmm_np(mean, stdev) == pytest.approx(expected, 0.2)

def test_init_params_np():
    # Arrange
    gmm_means = [2.9758800521512385]
    gmm_stdevs = [2.808703462976363]
    gmm_thetas = [1.0]
    expected = ([1.8025603679160627, 0.3772277583175714], 1)

    # Act
    output = _init_params_np(gmm_means, gmm_stdevs, gmm_thetas)

    # Assert
    assert int(gmm_thetas[0]) == output[1]
    assert output[0] == pytest.approx(expected[0], 0.2)

@pytest.mark.parametrize(
    "num_comp, expected",
    [
        (1, [(1e-08, np.inf), (1e-08, np.inf)]),
        (2, [(1e-08, np.inf), (1e-08, np.inf), 
             (1e-08, np.inf), (1e-08, np.inf), 
             (1e-08, 1)]),
        (3, [(1e-08, np.inf), (1e-08, np.inf), 
             (1e-08, np.inf), (1e-08, np.inf), 
             (1e-08, np.inf), (1e-08, np.inf), 
             (1e-08, 1), (1e-08, 1)])

    ]
)
def test_param_bounds(num_comp, expected):
    assert _param_bounds(num_comp) == expected

@pytest.mark.parametrize(
    "args, num_comp, expected",
    [
        ((1, 0.2), 1, (1, 4.0)),
        ((1, 0.2, 2, 0.5, 0.4), 2, (1, 4.0, 2, 1.0, 0.4)),
        ((4, 0.2, 6, 0.5, 7, 0.2, 0.3, 0.6), 3, (4, 4.0, 6, 1.0, 7, 4.0, 0.3, 0.6)) 
    ]
)
def test_convert_np_ab(args, num_comp, expected):
    assert _convert_np_ab(args, num_comp) == expected

@pytest.mark.parametrize(
    "args, data_vector, num_comp, expected",
    [
        ((1, 0.2), [1, 2, 3], 1, 11.29),
        ((1, 0.2, 2, 0.5, 0.4), [1, 2, 3], 2, 6.96),
        ((4, 0.2, 6, 0.5, 7, 0.2, 0.3, 0.6), [1, 2, 3], 3, 5.15)
    ]
)
def test_log_like_nb(args, data_vector, num_comp, expected):
    assert _log_like_nb(args, data_vector, num_comp) == pytest.approx(expected, 0.2)

def test_theta_constr_mle_2():
    # Arrange
    args = (1, 0.2, 2, 0.5, 0.4)
    expected = 0.6

    # Act
    output = _theta_constr_mle_2(args)

    # Assert
    assert output == expected

def test_theta_constr_mle_3():
    # Arrange
    args = (4, 0.2, 6, 0.5, 7, 0.2, 0.3, 0.6)
    expected = 0.1
    
    # Act
    output = _theta_constr_mle_3(args)

    # Assert
    assert output == pytest.approx(expected, 0.2)
    
@pytest.mark.parametrize(
    "args, data_vector, param_bounds, num_comp, expected",
    [
        ((1, 0.2), [1,2,3], [(1e-08, np.inf), (1e-08, np.inf)], 1, [14.85, 0.13]),
        ((1, 0.2, 2, 0.5, 0.4), [1, 2, 3], [(1e-08, np.inf), (1e-08, np.inf), (1e-08, np.inf), (1e-08, np.inf), (1e-08, 1)], 2, [3.21, 0.55, 15.00, 0.13, 1e-08]),
        ((4, 0.2, 6, 0.5, 7, 0.2, 0.3, 0.6), [1, 2, 3], 
            [(1e-08, np.inf), (1e-08, np.inf), 
             (1e-08, np.inf), (1e-08, np.inf), 
             (1e-08, np.inf), (1e-08, np.inf), 
             (1e-08, 1), (1e-08, 1)], 3, 
             [3.88, 0.64, 6.01, 0.32, 7.00, 0.29, 1e-08, 0.52])
    ]
)
def test_mle_mix_nb(mocker, args, data_vector, param_bounds, num_comp, expected):
    # Arrange
    mock_constr_2 = mocker.patch("src.immunopheno.models._theta_constr_mle_2",
                                 return_value=0.6)
    
    mock_constr_3 = mocker.patch("src.immunopheno.models._theta_constr_mle_3",
                                 return_value=0.1)

    # Act
    output = _mle_mix_nb(args, data_vector, param_bounds, num_comp).x

    # Assert
    assert list(output) == pytest.approx(expected, 0.2)

@pytest.mark.parametrize(
    "args, num_comp, expected",
    [
       ((14.85, 0.13), 1, (14.85, 0.88)),
       ((3.21, 0.55, 15.00, 0.13, 1e-08), 2, (3.21, 0.64, 15.0, 0.88, 1e-08)),
       ((3.88, 0.64, 6.01, 0.32, 7.00, 0.29, 1e-08, 0.52), 3, (3.88, 0.60, 6.01, 0.75, 7.0, 0.77, 1e-08, 0.52))
    ]
)
def test_convert_ab_np(args, num_comp, expected):
    assert _convert_ab_np(args, num_comp) == pytest.approx(expected, 0.2)

def test_aic_nb():
    # Arrange
    log_like = 100
    num_param = 2
    expected = -196

    # Act
    output = _aic_nb(log_like, num_param)

    # Assert
    assert output == expected

def test_nb_mle_results():
    # Arrange
    counts = [1, 2, 3]
    expected = {1: {'num_comp': 1,
                    'model': 'negative binomial (MLE)',
                    'aic': 12.65,
                    'nb_n_params': [400000.00],
                    'nb_p_params': [0.99],
                    'nb_thetas': [1]},
                2: {'num_comp': 2,
                    'model': 'negative binomial (MLE)',
                    'aic': 30.65,
                    'nb_n_params': [225010.00, 899999.99],
                    'nb_p_params': [0.85, 0.99],
                    'nb_thetas': [0.66]},
                3: {'num_comp': 3,
                    'model': 'negative binomial (MLE)',
                    'aic': 32.86,
                    'nb_n_params': [99999.99, 899999.99, 399999.99],
                    'nb_p_params': [0.70, 0.99, 0.93],
                    'nb_thetas': [0.25, 0.48]}}
    # Act
    output = _nb_mle_results(counts)

    # Assert
    assert expected[3]['aic'] == pytest.approx(output[3]['aic'], 0.2)