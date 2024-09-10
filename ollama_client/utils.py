import string
import traceback
from random import choices

def calculate_completion_runtime(func):
    func()

def print_stacktrace():
    print(traceback.format_exc())

def create_random_string(length: int) -> str:
    return ''.join(choices(string.ascii_uppercase + string.digits, k=length))