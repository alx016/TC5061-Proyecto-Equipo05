import re
from pathlib import Path
import tomllib
import unittest

import yaml


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

    def test_python_env_matches_pyproject(self):
        root = Path(__file__).resolve().parent.parent

        # loads the pinned dependencies of pyproject.toml
        with open(root / "pyproject.toml", "rb") as f:
            pyproject_deps = set(tomllib.load(f)["project"]["dependencies"])

        # loads the dependencies of python_env.yaml, without the line that installs our own package
        env = yaml.safe_load((root / "python_env.yaml").read_text(encoding="utf-8"))
        env_deps = set(env["dependencies"]) - {"-e ."}

        # checks that both match
        self.assertSetEqual(env_deps, pyproject_deps)


if __name__ == "__main__":
    unittest.main()
