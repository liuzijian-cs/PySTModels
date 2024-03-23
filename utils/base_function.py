import os

# function list:
# |- print_log(args, string) : Output logs to console and write to log file.
# |- print_log_args(args) : Output args to console and write to log file.

def print_log(args, string):
    """
    Output the input string to a log file and to the console via the print() function.
    """
    log_file = open(args.log_file, 'a')
    log_file.write(string + '\n')
    log_file.flush()
    print(string)

def print_log_args(args):
    """
    Output the arguments to the console via the print() function
    :param args:
    :return:
    """
    print(args)

