import logging
import os
from pathlib import Path
# from util.dist_utils import is_main_process


def _is_main_rank():
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank) == 0
    except ValueError:
        return rank in ("", "0")


def setup_logger(output_dir, level=logging.INFO):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("C2R")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    is_main_rank = _is_main_rank()
    logger.setLevel(level if is_main_rank else logging.WARNING)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)5s] [%(pathname)50s:%(lineno)d]\t| %(message)s', 
        '%Y-%m-%d %H:%M:%S'
    )

    if is_main_rank:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level=level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_debug_handler = logging.FileHandler(output_dir / 'debug.log', mode='a')
        file_debug_handler.setLevel(logging.DEBUG)
        file_debug_handler.setFormatter(formatter)
        logger.addHandler(file_debug_handler)

    # ERROR 레벨 이상의 로그를 `error.log`에 출력하는 Handler
    file_error_handler = logging.FileHandler(output_dir / 'error.log', mode='a')
    file_error_handler.setLevel(logging.ERROR)
    file_error_handler.setFormatter(formatter)
    logger.addHandler(file_error_handler)

    # logging.basicConfig(
    #     level=level if is_main_process() else logging.WARN,
    #     # format="%(asctime)s [%(levelname)s] %(message)s",
    #     format="%(asctime)s [%(levelname)7s]in [%(funcName)20s() in %(pathname)50s:%(lineno)3d]\t| %(message)s",
    #     handlers=[
    #         logging.StreamHandler(),
    #         logging.FileHandler(os.path.join(output_dir, "ywjang_log.txt"), mode="a", level=logging.DEBUG),
    #     ],
    # )

def get_logger(name="C2R"):
    return logging.getLogger(name)
