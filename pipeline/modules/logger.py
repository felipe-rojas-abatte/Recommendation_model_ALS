from library.packages import *

def setup_logger(log_file):
    # Create a logger object
    logger = logging.getLogger('recommendation_system')
    logger.setLevel(logging.INFO)
    
    # Create a file handler to write logs to the file
    file_handler = logging.FileHandler(log_file)
    
    # Create a formatter for the logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    return logger

log_file = 'recommendation_system.log'
logger = setup_logger(log_file)