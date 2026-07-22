from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class WorkspacePathTest(unittest.TestCase):
    def test_local_container_mount_uses_current_checkout(self):
        readme = (ROOT / "README.md").read_text()

        self.assertIn("$PWD:/workspace", readme)
        self.assertNotIn("/home/ywjang/LBA_LAVIS_uncertainty_v2", readme)

    def test_remote_output_root_is_configurable(self):
        script = (ROOT / "sync_output.sh").read_text()

        self.assertIn("LBA_REMOTE_ROOT", script)
        self.assertNotIn("/home/ywjang/LBA_Uv2", script)

    def test_remote_igvlm_root_is_configurable(self):
        script = (ROOT / "sync_output.sh").read_text()

        self.assertIn("IGVLM_REMOTE_ROOT", script)
        self.assertNotIn("/home/ywjang/IG-VLM", script)


if __name__ == "__main__":
    unittest.main()
