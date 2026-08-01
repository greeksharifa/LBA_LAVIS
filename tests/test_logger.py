import io
import logging
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from util.logger import get_logger, setup_logger


class LoggerTests(unittest.TestCase):
    def tearDown(self):
        logger = get_logger()
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

    def test_rank_zero_info_is_emitted_once_without_handler_duplication(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {}, clear=True
        ):
            output_dir = Path(tmp)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                setup_logger(output_dir, level=logging.INFO)
                setup_logger(output_dir, level=logging.INFO)
                logger = get_logger()
                self.assertEqual(logging.INFO, logger.level)
                logger.info("one-info-record")

            self.assertEqual(1, stderr.getvalue().count("one-info-record"))
            self.assertEqual(
                1,
                (output_dir / "debug.log").read_text().count("one-info-record"),
            )

    def test_non_main_rank_suppresses_routine_info_handlers(self):
        for env in ({"RANK": "1"}, {"LOCAL_RANK": "1"}):
            with self.subTest(env=env), tempfile.TemporaryDirectory() as tmp, patch.dict(
                "os.environ", env, clear=True
            ):
                output_dir = Path(tmp)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    setup_logger(output_dir, level=logging.INFO)
                    get_logger().info("rank-one-info")

                self.assertNotIn("rank-one-info", stderr.getvalue())
                debug_path = output_dir / "debug.log"
                if debug_path.exists():
                    self.assertNotIn("rank-one-info", debug_path.read_text())


if __name__ == "__main__":
    unittest.main()
