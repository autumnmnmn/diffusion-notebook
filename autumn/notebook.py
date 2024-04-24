"""
Jupyter-specific stuff, mostly
"""
import time
import uuid
import os.path
from datetime import date
from pathlib import Path
from IPython.core.magic import register_cell_magic

print("notebook reloaded")

def index_where(predicate, _list):
    return next(filter(lambda _tuple: predicate(_tuple[1]), enumerate(_list)))[0] 

def is_comment(line):
    return next(filter(lambda c: c != ' ', line)) == '#'

@register_cell_magic
def settings(args, cell):
    original_lines = [line for line in cell.split("\n")]
    i = index_where(lambda line: line.startswith("#!#"), original_lines)

    settings_id = uuid.uuid4()
    id_line = f"settings_id = \"{settings_id}\"" 
    
    if os.path.isfile("settings.py"):
        with open("settings.py", "r") as reader:
            private_lines = [line for line in reader.read().split("\n")]
        lines = [id_line] + private_lines + original_lines[i+1:]
    else:
        lines = [id_line] + original_lines[:i] + original_lines[i+1:]

    is_meaningful = lambda line: line != "" and not is_comment(line)
    
    log_content = "\n".join(filter(is_meaningful, lines))

    daily_directory = date.today().strftime("%d.%b.%Y")
    daily_directory = f"out/{daily_directory}"
    
    Path(daily_directory).mkdir(exist_ok=True, parents=True)

    with open(f"{daily_directory}/{settings_id}.settings", "w") as file:
        file.write(log_content)

    run_content = log_content + f"\ndaily_directory = \"{daily_directory}\""
    
    get_ipython().run_cell(run_content)

def file_snapshot(filename):
    try:
        with open(filename, "r") as reader:
            snapshot = reader.read()
        return f"# Private settings:\n\n{snapshot}\n\n"
    except:
        return "# No private settings.\n\n"

def _gc():
    gc.collect()
    torch.cuda.empty_cache()

class Timer(object):
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *args):
        print(f"{self.name}: {time.perf_counter() - self.t}")