"""Regression checks for actual comparison figures and their CLI."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import plot_comparison as comparison


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["episode", "delivery_rate", "average_hops", "average_delay"]


def write_csv(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "TG_MAPPO_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return path


class ComparisonPlotTests(unittest.TestCase):
    def capture_figures(self, directory, rows):
        source = write_csv(directory / "logs", rows)
        original = source.read_bytes()
        savefig = comparison.plt.savefig
        figures = []

        def capture(path, **kwargs):
            figures.append([
                (line.get_xdata().tolist(), line.get_ydata().tolist(), line.get_marker())
                for line in comparison.plt.gca().lines])
            savefig(path, **kwargs)

        with patch.object(comparison.plt, "savefig", side_effect=capture):
            paths = comparison.plot_all(str(source.parent), str(directory / "plots"))
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(Path(path).stat().st_size > 1000 for path in paths))
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(comparison.plt.get_fignums(), [])
        return figures

    def test_single_episode_has_visible_markers_in_each_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            figures = self.capture_figures(Path(temp), [[0, 0.5, 2, 3]])
        for figure, value in zip(figures, (0.5, 2.0, 3.0)):
            x, y, marker = figure[0]
            self.assertEqual((x, y), ([0], [value]))
            self.assertNotIn(marker, (None, "None", "", " "))

    def test_multiple_episodes_keep_all_real_values(self):
        with tempfile.TemporaryDirectory() as temp:
            figures = self.capture_figures(Path(temp), [[0, 0.5, 2, 3], [2, 0.75, 4, 5]])
        for figure, expected in zip(figures, ([0.5, 0.75], [2, 4], [3, 5])):
            self.assertEqual(figure[0][:2], ([0, 2], expected))

    def test_empty_input_directory_fails_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaisesRegex(ValueError, "No .*metrics.csv"):
                comparison.plot_all(str(directory), str(directory / "plots"))
            self.assertFalse((directory / "plots").exists())

    def test_header_only_csv_is_rejected_without_creating_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = write_csv(directory / "logs", [])
            with self.assertRaisesRegex(ValueError, "no metric rows"):
                comparison.plot_all(str(source.parent), str(directory / "plots"))
            self.assertFalse((directory / "plots").exists())

    def test_nonfinite_metrics_report_file_and_row(self):
        with tempfile.TemporaryDirectory() as temp:
            for field in range(1, 4):
                for value in ("nan", "inf", "-inf"):
                    row = [0, 0.5, 2, 3]
                    row[field] = value
                    source = write_csv(Path(temp), [row])
                    with self.subTest(field=field, value=value), self.assertRaises(ValueError) as caught:
                        comparison.read_metrics(str(source))
                    self.assertIn(str(source) + ":2", str(caught.exception))
                    self.assertIn("finite", str(caught.exception))

    def test_malformed_numbers_and_missing_cells_report_the_row(self):
        with tempfile.TemporaryDirectory() as temp:
            for row in (["not-an-episode", 0.5, 2, 3], [0, "bad", 2, 3], [0, 0.5, 2]):
                source = write_csv(Path(temp), [row])
                with self.subTest(row=row), self.assertRaises(ValueError) as caught:
                    comparison.read_metrics(str(source))
                self.assertIn(str(source) + ":2", str(caught.exception))
                self.assertIn("invalid metric value", str(caught.exception))

    def test_zero_metrics_are_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            source = write_csv(Path(temp), [[0, 0, 0, 0]])
            self.assertEqual(comparison.read_metrics(str(source)), {
                "episode": [0], "delivery_rate": [0.0],
                "average_hops": [0.0], "average_delay": [0.0]})

    def test_cli_invalid_data_fails_without_pngs_or_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = write_csv(directory / "moves_5" / "logs", [[0, "nan", 2, 3]])
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts/plot_comparison.py"),
                "--output-root", str(directory), "--num-moving-buoys", "5"],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=30)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(str(source) + ":2", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertFalse((directory / "moves_5/comparison_plots").exists())

    def test_cli_creates_three_single_episode_pngs(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_csv(directory / "moves_5" / "logs", [[0, 0.5, 2, 3]])
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts/plot_comparison.py"),
                "--output-root", str(directory), "--num-moving-buoys", "5"],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(len(list((directory / "moves_5/comparison_plots").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
