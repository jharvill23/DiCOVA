import os
from easydict import EasyDict as edict
import yaml
from config import get_config




def main():
    # cwd = os.getcwd()
    os.chdir('..')  # set working directory to root if you run this script by itself
    config = get_config.get()
    stop = None

if __name__ == "__main__":
    main()