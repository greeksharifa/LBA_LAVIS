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

import json

default = {
    "init_prompt": "You have to answer the main-question: '{}'. You can use the following Q&A results to help you answer: ",
    "pair_prompt": "{} {}. ",
    "final_prompt": "The question \"{}\" can be answered using the image. A short answer is "
  }


while True:
    print(Colors.BRIGHT_MAGENTA + "new prompt: enter 'new'. quit: 'quit'. " + Colors.RESET)
    inputs = input()
    if inputs == "quit":
        break
    elif inputs == "new":
        new_prompt = dict(default)
        
        
        def prompt_process(_prompt_type):
            print(
                Colors.BRIGHT_MAGENTA + f"if you want to pass, just enter 'pass'. Enter {_prompt_type}: " + Colors.RESET)
            _prompt = input()
            if _prompt == "pass":
                return None
            else:
                return _prompt
        
        
        for prompt_type in ["init_prompt", "pair_prompt", "final_prompt"]:
            new_prompt.update({prompt_type: prompt_process(prompt_type)})
        
        print(Colors.BRIGHT_YELLOW + "new_prompt: " + json.dumps(new_prompt, indent=4) + Colors.RESET)
        print(Colors.BRIGHT_MAGENTA + "Are you sure? Enter 'yes' or 'no'." + Colors.RESET)
        _yes_or_no = input()
        if _yes_or_no == "yes":
            print('run!')
        else:
            print('pass!')
            pass

print(Colors.YELLOW + "Evaluation finished." + Colors.RESET)
