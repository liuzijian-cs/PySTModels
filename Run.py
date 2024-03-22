import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  # Create the argument parser

    # Device:
    parser.add_argument('--device', type=str, default='cuda:0',help='cuda or cpu')





    args = parser.parse_args()