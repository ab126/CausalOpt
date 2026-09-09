"""Regression coverage for the derivatives loader and explicit complete pairing."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from causal_opt.fmri_data import (
    build_multisubject_lagged_data,
    build_static_fmri_matrix,
    load_paired_tsv_states,
    load_tsv_state,
    validate_paired_states,
)


class TestTSVStates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # An ancestor resembling a subject must not determine subject identity.
        self.root = Path(self.tmp.name) / "sub-999" / "rois"
        self.root.mkdir(parents=True)
        self.labels = self.root / "labels.json"
        self.labels.write_text(json.dumps({
            "1": {"roi_name": "A", "centers_mni": [[1, 2, 3]]},
            "2": {"roi_name": "B", "centers_mni": [[4, 5, 6]]},
        }), encoding="utf-8")

    def write(self, subject, state, text="B\tA\n1\t2\n3\t4\n", run=""):
        folder = self.root / f"sub-{subject}" / f"ses-{state}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"sub-{subject}_ses-{state}{run}_timeseries.tsv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_string_paths_numeric_order_and_tsv_coordinates(self):
        self.write("10", "control")
        self.write("2", "control", "B\tA\n5\t6\n7\t8\n9\t10\n")
        data = load_tsv_state(str(self.root), str(self.labels), "control", "001")
        self.assertEqual(data.subject_ids, ("Subject002", "Subject010"))
        self.assertEqual(data.roi_names, ("B", "A"))
        np.testing.assert_array_equal(data.roi_xyz, [[4, 5, 6], [1, 2, 3]])
        self.assertEqual(build_static_fmri_matrix(data).shape, (5, 2))
        current, lagged, _ = build_multisubject_lagged_data(data, 1, center=False)
        np.testing.assert_array_equal(current[:, 0], [7, 9, 3])
        np.testing.assert_array_equal(lagged[0, :, 0], [5, 7, 1])

    def test_missing_control_is_strict_by_default_and_explicitly_reported(self):
        self.write("002", "control")
        self.write("002", "sdv")
        self.write("015", "sdv")
        with self.assertRaisesRegex(ValueError, "missing control=.*Subject015"):
            load_paired_tsv_states(self.root, self.labels)
        original = load_tsv_state(self.root, self.labels, "sdv")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            control, sdv = load_paired_tsv_states(self.root, self.labels, pairing="intersection")
        self.assertEqual(len(caught), 1)
        self.assertIn("Subject015", str(caught[0].message))
        self.assertEqual(control.subject_ids, ("Subject002",))
        self.assertEqual(sdv.subject_ids, control.subject_ids)
        self.assertIn("Subject015", original.timeseries)
        validate_paired_states(control, sdv)

    def test_no_overlap_and_invalid_policy(self):
        self.write("002", "control")
        self.write("015", "sdv")
        with self.assertRaisesRegex(ValueError, "no paired subjects"):
            load_paired_tsv_states(self.root, self.labels, pairing="intersection")
        with self.assertRaisesRegex(ValueError, "pairing must"):
            load_paired_tsv_states(self.root, self.labels, pairing="drop")

    def test_duplicate_runs_cannot_overwrite(self):
        self.write("002", "control", run="_run-1")
        self.write("002", "control", run="_run-2")
        with self.assertRaisesRegex(ValueError, "Multiple timeseries"):
            load_tsv_state(self.root, self.labels, "control")

    def test_invalid_timeseries(self):
        for text, message in [
            ("B\tB\n1\t2\n", "duplicate"),
            ("B\tC\n1\t2\n", "ROI labels differ"),
            ("B\tA\n", "Empty"),
            ("B\tA\nNaN\t2\n", "Non-finite"),
            ("B\tA\ninf\t2\n", "Non-finite"),
            ("B\tA\ntext\t2\n", "Invalid numeric"),
            ("B\tA\n1\t2\t3\n", "Malformed TSV row"),
            ("B\tA\n1\n", "Malformed TSV row"),
        ]:
            with self.subTest(text=text):
                self.write("002", "control", text)
                with self.assertRaisesRegex(ValueError, message):
                    load_tsv_state(self.root, self.labels, "control")

    def test_inconsistent_roi_order(self):
        self.write("002", "control")
        self.write("003", "control", "A\tB\n1\t2\n")
        with self.assertRaisesRegex(ValueError, "ROI order differs"):
            load_tsv_state(self.root, self.labels, "control")

    def test_cross_state_atlas_mismatch_still_fails_intersection(self):
        self.write("002", "control")
        self.write("002", "sdv", "A\tB\n1\t2\n")
        with self.assertRaisesRegex(ValueError, "ROI ordering differs"):
            load_paired_tsv_states(self.root, self.labels, pairing="intersection")

    def test_bad_coordinates_and_duplicate_labels(self):
        self.write("002", "control")
        labels = json.loads(self.labels.read_text())
        labels["1"]["centers_mni"] = [[1, 2]]
        self.labels.write_text(json.dumps(labels))
        with self.assertRaisesRegex(ValueError, "finite 3D MNI center"):
            load_tsv_state(self.root, self.labels, "control")
        labels["1"]["centers_mni"] = [[1, 2, 3]]
        labels["2"]["roi_name"] = "A"
        self.labels.write_text(json.dumps(labels))
        with self.assertRaisesRegex(ValueError, "Duplicate ROI label"):
            load_tsv_state(self.root, self.labels, "control")

    def test_session_and_filename_identity(self):
        path = self.write("002", "control")
        with self.assertRaisesRegex(ValueError, "must map"):
            load_tsv_state(self.root, self.labels, "control", "002")
        path.rename(path.with_name("sub-003_ses-control_timeseries.tsv"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_tsv_state(self.root, self.labels, "control")

    def test_no_files(self):
        with self.assertRaisesRegex(FileNotFoundError, "session 'sdv'"):
            load_tsv_state(self.root, self.labels, "sdv")


if __name__ == "__main__":
    unittest.main()
