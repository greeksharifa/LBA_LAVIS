import logging

from lavis.common import dist_utils


class Colors:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    
    RESET = '\033[0m'
    
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    BRIGHT_END = '\033[0m'
    
def print_color(msg, color=Colors.BRIGHT_MAGENTA):
    logging.info(color + msg + Colors.RESET)


def print_sample(samples, output_text="", msg="print sample...", color=Colors.BRIGHT_MAGENTA):
    # if dist_utils.is_main_process():
    logging.info(color + msg + Colors.RESET)
    for key in samples.keys():
        if key != "image":
            print_color = Colors.BRIGHT_YELLOW if key == "prompt" else color
            logging.info(print_color + f"key: {key},\tvalue: {samples[key]}" + Colors.RESET)
    if output_text != "":
        if isinstance(output_text, list):
            for text in output_text:
                logging.info(Colors.BRIGHT_YELLOW + "output_text: " + text + Colors.RESET)
        # logging.info(color + output_text + Colors.RESET)
                
                