"""Behavioral checks for research baselines and frozen spatial transforms."""
import unittest

import numpy as np
from sklearn.base import clone

from experiments.spatial import CSP, FBCSP, spatial_models


class SpatialBaselineTests(unittest.TestCase):
    @staticmethod
    def signals(seed, count=30):
        rng = np.random.default_rng(seed)
        labels = np.arange(count) % 2
        data = rng.normal(size=(count, 6, 321)) * 1e-6
        oscillation = np.sin(2 * np.pi * 10 * np.arange(321) / 160)
        for i, label in enumerate(labels):
            data[i, label] += oscillation * 8e-6
        return data, labels

    def test_spatial_models_learn_known_signal_on_unseen_trials(self):
        train, labels = self.signals(1)
        test, expected = self.signals(2)
        for name, model in spatial_models(160).items():
            with self.subTest(model=name):
                fitted = clone(model).fit(train, labels)
                self.assertGreaterEqual(fitted.score(test, expected), .9)

    def test_transform_does_not_refit_and_selection_is_repeatable(self):
        train, labels = self.signals(1)
        test, _ = self.signals(2)
        model = FBCSP().fit(train, labels)
        filters = [csp.filters_.copy() for _, csp in model.banks_]
        support = model.selector_.get_support().copy()
        features = model.transform(test)
        np.testing.assert_array_equal(support, model.selector_.get_support())
        for before, (_, csp) in zip(filters, model.banks_):
            np.testing.assert_array_equal(before, csp.filters_)
        repeated = clone(model).fit(train, labels).transform(test)
        np.testing.assert_array_equal(features, repeated)

    def test_invalid_component_count_and_single_class_are_rejected(self):
        train, labels = self.signals(1)
        with self.assertRaisesRegex(ValueError, 'components'):
            CSP(n_components=3).fit(train, labels)
        with self.assertRaisesRegex(ValueError, 'two label classes'):
            CSP().fit(train, np.zeros_like(labels))


if __name__ == '__main__':
    unittest.main()
