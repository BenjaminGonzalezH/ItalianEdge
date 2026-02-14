"""Unit tests for ReadSolution file readers.

Purpose of this file:
- Validate normal and edge-case behavior of `read_solutions_file`.
- Ensure data shape/headers are parsed correctly across supported formats.
- Verify expected errors are raised for invalid inputs.
"""

######### Libraries #########
import unittest                     # Test framework.
import os                           # File creation/removal for fixtures.
import csv                          # CSV fixture generation.

from ParetoInsight_CPU.ReadSolution import read_solutions_file


class TestReadSolutionsFile(unittest.TestCase):
    """Test suite for successful reads, format handling, and error cases."""

    ########################## Test Initialization ##########################
    # Purpose:
    # - Build reusable input fixtures before each test.
    # - Keep tests independent by recreating files every run.
    def setUp(self):
        self.file_with_ids = "test_with_ids.csv"
        with open(self.file_with_ids, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SolID", "Gene1", "Gene2"])
            writer.writerow(["Sol1", 2, 3])
            writer.writerow(["Sol2", 2, 3])
            writer.writerow(["Sol3", 3, 3])

        self.file_without_ids = "test_without_ids.csv"
        with open(self.file_without_ids, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Gene1", "Gene2", "Gene3"])
            writer.writerow([1, 2, 3])
            writer.writerow([2, 2, 3])
            writer.writerow([3, 3, 3])

        self.empty_file = "empty.csv"
        with open(self.empty_file, "w") as f:
            pass

    ########################## Cleanup ##########################
    # Purpose:
    # - Remove temporary fixture files after each test.
    # - Prevent cross-test pollution and keep local workspace clean.
    def tearDown(self):
        for filename in [
            self.file_with_ids,
            self.file_without_ids,
            self.empty_file,
        ]:
            if os.path.exists(filename):
                os.remove(filename)

    ########################## Tests ##########################
    # Purpose:
    # - Verify expected behavior for valid files.
    # - Validate error handling for invalid or unsupported inputs.

    def test_read_with_ids(self):
        """Purpose: confirm CSV with identifier column is loaded without dropping columns."""
        matrix, genes = read_solutions_file(self.file_with_ids)

        self.assertEqual(genes, ["SolID", "Gene1", "Gene2"])
        self.assertEqual(matrix.shape, (3, 3))

    def test_read_without_ids(self):
        """Purpose: confirm standard numeric CSV is parsed with correct headers and values."""
        matrix, genes = read_solutions_file(self.file_without_ids)

        self.assertEqual(genes, ["Gene1", "Gene2", "Gene3"])
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix[0, 0], 1)

    def test_empty_file(self):
        """Purpose: confirm an empty CSV raises ValueError during DataFrame validation."""
        with self.assertRaises(ValueError):
            read_solutions_file(self.empty_file)

    def test_file_not_found(self):
        """Purpose: confirm missing file paths raise FileNotFoundError immediately."""
        with self.assertRaises(FileNotFoundError):
            read_solutions_file("non_existent.csv")

    def test_unsupported_format(self):
        """Purpose: confirm unsupported extensions raise ValueError."""
        fake_file = "test.json"
        with open(fake_file, "w") as f:
            f.write("{}")

        with self.assertRaises(ValueError):
            read_solutions_file(fake_file)

        os.remove(fake_file)

    def test_removes_unnamed_columns(self):
        """Purpose: confirm auto-generated 'Unnamed' columns are removed from output."""
        filename = "test_unnamed.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Unnamed: 0", "Gene1", "Gene2"])
            writer.writerow([0, 1, 2])
            writer.writerow([1, 3, 4])

        matrix, genes = read_solutions_file(filename)

        self.assertEqual(genes, ["Gene1", "Gene2"])
        self.assertEqual(matrix.shape, (2, 2))

        os.remove(filename)

    def test_semicolon_delimiter(self):
        """Purpose: confirm delimiter sniffing correctly reads semicolon-separated CSV files."""
        filename = "test_semicolon.csv"
        with open(filename, "w") as f:
            f.write("Gene1;Gene2;Gene3\n")
            f.write("1;2;3\n")
            f.write("4;5;6\n")

        matrix, genes = read_solutions_file(filename)

        self.assertEqual(genes, ["Gene1", "Gene2", "Gene3"])
        self.assertEqual(matrix.shape, (2, 3))

        os.remove(filename)

    def test_pickle_file(self):
        """Purpose: confirm pickled pandas DataFrame input is loaded correctly."""
        import pandas as pd

        filename = "test.pkl"
        df = pd.DataFrame({
            "Gene1": [1, 2],
            "Gene2": [3, 4]
        })
        df.to_pickle(filename)

        matrix, genes = read_solutions_file(filename)

        self.assertEqual(genes, ["Gene1", "Gene2"])
        self.assertEqual(matrix.shape, (2, 2))

        os.remove(filename)

    def test_pickle_not_dataframe(self):
        """Purpose: confirm pickled objects that are not DataFrames raise TypeError."""
        import pickle

        filename = "invalid.pkl"
        with open(filename, "wb") as f:
            pickle.dump([1, 2, 3], f)

        with self.assertRaises(TypeError):
            read_solutions_file(filename)

        os.remove(filename)

    def test_uppercase_extension(self):
        """Purpose: confirm extension matching is case-insensitive (e.g., '.CSV')."""
        filename = "test.CSV"
        with open(filename, "w") as f:
            f.write("Gene1,Gene2\n1,2\n")

        matrix, genes = read_solutions_file(filename)

        self.assertEqual(genes, ["Gene1", "Gene2"])

        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
