import logging
import os
import sys
from datetime import datetime

def get_log_file_path(module_name: str) -> str:
    # Define the base log directory and ensure it exists
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    current_time_stamp=datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    # Define a log file name based on the module name
    LOG_FILE_NAME = f"{module_name}_{current_time_stamp}.log"
    LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
    return LOG_FILE_PATH

def setup_logging(module_name: str):
    # Set up logging to use the module-specific log file
    log_file_path = get_log_file_path(module_name)
    logging.basicConfig(filename=log_file_path,
                        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                        level=logging.DEBUG)

# Extract the module name from sys.argv[0] or use __name__ for testing purposes
module_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
setup_logging(module_name)

# Create a logger for this module
logger = logging.getLogger(__name__)
