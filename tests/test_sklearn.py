import pytest

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from fracdiff import Fracdiff


class TestScikitLearn:
    @pytest.mark.parametrize("seed", [42])
    @pytest.mark.parametrize("n_samples", [20, 100])
    @pytest.mark.parametrize("n_features", [1, 10])
    @pytest.mark.parametrize("d", [0.5])
    def test_sample(self, seed, n_samples, n_features, d):
        np.random.seed(seed)

        X = np.random.randn(n_samples, n_features)
        fracdiff = Fracdiff(d).transform(X)

    @pytest.mark.parametrize("seed", [42])
    @pytest.mark.parametrize("n_samples", [20, 100])
    @pytest.mark.parametrize("n_features", [1, 10])
    @pytest.mark.parametrize("d", [0.5])
    def test_sample(self, seed, n_samples, n_features, d):
        np.random.seed(seed)

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("fracdiff", Fracdiff(d)),
                ("regressor", LinearRegression()),
            ]
        )

        pipeline.fit(X, y)
