import subprocess
import logging
import logging.config


logging.config.fileConfig(fname='log.conf', disable_existing_loggers=False)

# Get the logger specified in the file
logger = logging.getLogger(__name__)

logger.debug('This is a debug message')

subprocess.run(['python', 'feature_engineering.py'])

subprocess.run(['python', 'train.py'])