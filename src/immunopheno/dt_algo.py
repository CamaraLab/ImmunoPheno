import time

import numpy as np
import pandas as pd


class Algo(object):
    def __init__(self):
        self.data = pd.DataFrame()
        self.label = pd.DataFrame()
        self.feature_importance = pd.DataFrame()
        self.random_state = int(time.time())

    def k_features(self, k=None):
        """
        This function should be executed after fit.
        The function will return k important features.

        :param k: An integer to specify the desired number of features.
        :return: A list that includes k important features.
        """
        copy = self.feature_importance.copy()
        non_zero_indices = (self.feature_importance.iloc[:, 0] != 0)
        copy = copy[non_zero_indices]
        if k is not None and k < copy.shape[0]:
            return copy[0:k].index.tolist()
        else:
            return copy.index.tolist()