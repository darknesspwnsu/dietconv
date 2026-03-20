import unittest

from scripts.update_readme_benchmarks import (
    END_MARKER,
    START_MARKER,
    replace_digest_section,
)


class UpdateReadmeBenchmarksTests(unittest.TestCase):
    def test_replace_digest_section_replaces_only_marker_region(self) -> None:
        readme = "\n".join(
            [
                "# Title",
                START_MARKER,
                "old digest",
                END_MARKER,
                "tail",
            ]
        )

        updated = replace_digest_section(readme, "## Benchmark digest\n\nnew content")

        self.assertIn("## Benchmark digest\n\nnew content", updated)
        self.assertNotIn("old digest", updated)
        self.assertEqual(updated.count(START_MARKER), 1)
        self.assertEqual(updated.count(END_MARKER), 1)
        self.assertTrue(updated.endswith(f"{END_MARKER}\ntail"))


if __name__ == "__main__":
    unittest.main()
