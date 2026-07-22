from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).parents[1] / "research_tools" / "generate_subqa_chatgpt.py"


class ResearchToolCredentialTest(unittest.TestCase):
    def test_openai_key_is_read_from_environment_only(self):
        source = SCRIPT.read_text()

        self.assertIn('os.environ["OPENAI_API_KEY"]', source)
        self.assertNotIn("temp/openai_key.txt", source)
        self.assertNotIn("chatgpt_openai_api_key", source)

    def test_script_compiles_without_syntax_warnings(self):
        subprocess.check_call(
            [
                sys.executable,
                "-W",
                "error::SyntaxWarning",
                "-m",
                "py_compile",
                str(SCRIPT),
            ]
        )


if __name__ == "__main__":
    unittest.main()
