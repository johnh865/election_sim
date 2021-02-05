
import yaml
import shutil
import os
from logging import config

from votesim.definitions import DIR_MODULE, DIR_PROJECT





def start(config_name='logconfig.yaml'):
    """Start the module logger for debugging. 
    
    Configure the logger using logconfig.yaml found in project base
    directory.
    """
    
    
    template_path = os.path.join(DIR_MODULE, 
                                 'utilities',
                                 'log',
                                 'template.yaml')
    
    config_path = os.path.join(DIR_PROJECT, config_name)
    if not os.path.exists(config_path):
        shutil.copy(template_path, config_path)
        
    with open(config_path, 'rt') as f:
        curr = os.getcwd()
        config_data = yaml.safe_load(f.read())
        
        os.chdir(DIR_PROJECT)
        config.dictConfig(config_data)
        os.chdir(curr)
        