import os
import glob
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="image classification")
    parser.add_argument("--data_path", type=str, dest="data_path", help="the path of dataset")
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=20)
    args, _ = parser.parse_known_args()
    print(args.data_path)


    


if __name__ == "__main__":
    main()
