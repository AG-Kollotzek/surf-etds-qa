"""Raw data lookup across the campaigns of a linac."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_paths  # noqa: E402


class DataPathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.old = data_paths.CAMPAIGNS_DIR
        data_paths.CAMPAIGNS_DIR = self.tmp
        for campaign, names in {
            "2026-08-06_L3": ["phantom/ETsurface_easyQA_20260806_163851.csv",
                              "phantom/ETsurface_easyQA_20260806_163851_QA.csv",
                              "etd/TrackingResult_2026-08-06_16-43-56.json"],
            "2026-02-19_L3": ["phantom/ETD_QA_BasicPoP_20260219_120000.csv"],
            "2026-08-06_L4": ["phantom/ETsurface_easyQA_20260806_163851.csv"],
        }.items():
            for name in names:
                path = self.tmp / campaign / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x")

    def tearDown(self):
        data_paths.CAMPAIGNS_DIR = self.old
        shutil.rmtree(self.tmp)

    def test_finds_exactly_one_file_per_linac(self):
        self.assertEqual(data_paths.surf_csv("3", "163851").name, "ETsurface_easyQA_20260806_163851.csv")
        self.assertEqual(data_paths.surf_qa_csv("3", "163851").name, "ETsurface_easyQA_20260806_163851_QA.csv")
        self.assertEqual(data_paths.etd_json("3", "164356").name, "TrackingResult_2026-08-06_16-43-56.json")

    def test_other_linac_is_not_searched(self):
        self.assertEqual(data_paths.surf_csv("4", "163851").parent.parent.name, "2026-08-06_L4")

    def test_missing_and_ambiguous(self):
        with self.assertRaises(FileNotFoundError):
            data_paths.surf_csv("3", "999999")
        (self.tmp / "2026-02-19_L3" / "phantom" / "copy_163851.csv").write_text("x")
        with self.assertRaises(FileNotFoundError):
            data_paths.surf_csv("3", "163851")

    def test_stamp_format(self):
        self.assertEqual(data_paths.etd_stamp_dashed("161959"), "16-19-59")
        self.assertEqual(data_paths.etd_stamp_dashed("16-19-59"), "16-19-59")


if __name__ == "__main__":
    unittest.main()
