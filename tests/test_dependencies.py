import re
from pathlib import Path
import unittest


class TestProjectDependencies(unittest.TestCase):
    def test_dependencies_are_pinned(self):
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        self.assertTrue(pyproject_path.exists(), "pyproject.toml must exist")

        content = pyproject_path.read_text(encoding="utf-8")
        dep_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
        self.assertIsNotNone(dep_match, "dependencies block not found in pyproject.toml")

        raw_deps = re.findall(r"\"([^\"]+)\"", dep_match.group(1))
        self.assertGreater(len(raw_deps), 0, "No dependencies declared in pyproject.toml")

        unpinned = [dep for dep in raw_deps if "==" not in dep]
        self.assertListEqual(
            unpinned,
            [],
            f"Floating dependencies found without exact version pinning ('=='): {unpinned}",
        )


if __name__ == "__main__":
    unittest.main()
