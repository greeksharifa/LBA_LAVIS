import configparser
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).parents[1]


class SubmoduleConfigTest(unittest.TestCase):
    def test_every_gitlink_has_a_submodule_mapping(self):
        config = configparser.ConfigParser()
        config.read(ROOT / ".gitmodules")
        mapped_paths = {config[section]["path"] for section in config.sections()}

        output = subprocess.check_output(
            ["git", "ls-files", "-s"], cwd=ROOT, text=True
        )
        gitlinks = {
            line.split(maxsplit=3)[3]
            for line in output.splitlines()
            if line.startswith("160000 ")
        }

        self.assertEqual(gitlinks, mapped_paths & gitlinks)


if __name__ == "__main__":
    unittest.main()
