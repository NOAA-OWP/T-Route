import logging
import sys   
from datetime import datetime
import os

def log_level_set(input_parameters):
    '''
    Set logging level and specify logger configuration.
    
    Arguments
    ---------
    input_parameters (dict): User input logging parameters
    
    Returns
    -------
    None
    
    Notes
    -----
    In the absense of user-specified logging level, level defaults to DEGUG
    See also https://docs.python.org/3/library/logging.html
    
    '''
    log_level = input_parameters.get("log_level", 'DEBUG')
    if input_parameters.get('log_directory'):
        directory = input_parameters.get('log_directory')
        if not os.path.exists(directory):
            os.makedirs(directory)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_name = f"LOG_{timestamp}.log"
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s]: %(message)s',
            handlers=[
            logging.FileHandler(os.path.join(directory, log_file_name), mode='w'),  # Log to a file
            logging.StreamHandler(sys.stdout)  
        ])
    else:       
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s]: %(message)s',
            stream=sys.stderr,
        )   
