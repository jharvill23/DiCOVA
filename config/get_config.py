from easydict import EasyDict as edict
import yaml
import os

print('hey')

def get():
    # cwd = os.getcwd()
    config = edict(yaml.load(open(os.path.join('config', 'config.yml')), Loader=yaml.SafeLoader))
    return config
