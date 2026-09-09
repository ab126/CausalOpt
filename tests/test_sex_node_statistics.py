import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch

from causal_opt.fmri_inference import compute_sex_node_statistics
from causal_opt.fmri_plotting import plot_sex_node_reorganization


class SexNodeStatisticsTests(unittest.TestCase):
    def test_node_test_scores_before_subtraction_and_matches_valid_indices(self):
        male = np.array([[0., -2.], [0., 0.]])
        female = -male
        def boot(matrix, extra):
            return {"records": [
                {"index": 3, "valid": True, "delta_W": matrix},
                {"index": extra, "valid": True, "delta_W": matrix * 10},
                {"index": 8, "valid": False, "delta_W": None},
            ]}
        stats = compute_sex_node_statistics(
            male, female, boot(male, 4), boot(female, 5), ["A", "B"],
        )
        # Opposite edge signs give equal reorganization, not abs(female - male).
        assert np.all(stats.female_minus_male == 0)
        assert np.all(stats.p_boot == 1)
        assert np.all(stats.q_fdr == 1)
        assert np.all(stats.valid_bootstrap_pairs == 1)


    def test_node_centered_bootstrap_tail_and_no_valid_pairs(self):
        zero = np.zeros((2, 2))
        edge = np.array([[0., 2.], [0., 0.]])
        male = {"records": [{"index": i, "valid": True, "delta_W": zero} for i in range(3)]}
        female = {"records": [{"index": i, "valid": True, "delta_W": edge} for i in range(3)]}
        stats = compute_sex_node_statistics(zero, edge, male, female, ["A", "B"])
        assert np.all(stats.p_boot == .25)
        assert np.all(stats.q_fdr == .25)
        with self.assertRaisesRegex(ValueError, "No bootstrap"):
            compute_sex_node_statistics(zero, edge, male, {"records": []}, ["A", "B"])


    def test_plot_stars_follow_roi_labels_and_connector_midpoints(self):
        from matplotlib.figure import Figure
        names = ["A", "B", "C", "D"]
        male = pd.DataFrame({"roi": names, "total_change": [1., 4., 2., 3.]})
        female = pd.DataFrame({"roi": names, "total_change": [2., 5., 3., 4.]})
        # Deliberately different order; strict boundaries: .01 gets *, .05 gets none.
        stats = pd.DataFrame({"roi": ["D", "B", "A", "C"], "p_boot": [.05, .009, .01, .0009]})
        figures = []
        with patch.object(Figure, "savefig", lambda fig, *a, **k: figures.append(fig)):
            plot_sex_node_reorganization(male, female, "unused.png", node_statistics=stats)
        markers = {t.get_text(): t.get_position() for t in figures[0].axes[0].texts}
        assert markers == {"*": (1.5, 0), "***": (2.5, 1), "**": (4.5, 3)}


if __name__ == "__main__":
    unittest.main()
