import os
import wget
import zipfile

#LOGGING
import logging
import time
import sys
# LOGGING SETUP
logger = logging.getLogger('get dataset')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(filename)s: %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
logging.Formatter.converter = time.gmtime
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Download the zipped dataset
url = 'https://drive.google.com/uc?export=download&id=1oyRfNSV2AnlvK4_qv9jgEHUXVOpP4y22'
zip_name = "data.zip"
wget.download(url, zip_name)

# Unzip it and standardize the .csv filename
with zipfile.ZipFile(zip_name, "r") as zip_ref:
	zip_ref.extractall()

os.remove(zip_name)
logger.info('save dataset done')