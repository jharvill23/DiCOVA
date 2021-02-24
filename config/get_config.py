from easydict import EasyDict as edict
import yaml
import os

def get():
    config = edict(yaml.load(open(os.path.join('config', 'config.yml')), Loader=yaml.SafeLoader))
    return config

def path():
    return os.path.abspath(os.path.join('config', 'config.yml'))
