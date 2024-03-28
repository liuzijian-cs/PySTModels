import os
import re


# function list:
# |- print_log(args, string) : Output logs to console and write to log file.
# |- print_log_args(args) : Output args to console and write to log file.


class Color:
    K = '\033[30m'  # 黑色 BLACK
    R = '\033[31m'  # 红色 Red
    G = '\033[32m'  # 绿色 Green
    B = '\033[34m'  # 蓝色 Blue
    Y = '\033[33m'  # 黄色 Yellow
    P = '\033[35m'  # 紫色 Purple
    C = '\033[36m'  # 青色 Cyan
    W = '\033[37m'  # 白色 White
    RE = '\033[0m'  # RESET


def print_log(args, string, recursion_depth=1):
    """
    Output the input string to a log file and to the console via the print() function.
    """
    cleaned_string = (' ' * (recursion_depth * 2)) + re.sub(r'\x1b\[[0-9;]*m', '', string)  # 删除转义字符

    log_file = open(args.log_file, 'a')
    log_file.write(cleaned_string + '\n')
    log_file.flush()
    print(string)


def print_log_args(args):
    """
    Output the arguments to the console via the print() function
    :param args:
    :return:
    """
    print(args)
