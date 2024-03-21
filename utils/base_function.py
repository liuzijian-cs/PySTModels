import os


def print_log(args, string):
    """
    Output the input string to a log file and to the console via the print() function.
    """
    log_file = open(args.log_file, 'a')
    log_file.write(string + '\n')
    log_file.flush()
    print(string)
