import logging
import time
from pathlib import Path
import sys
from os.path import dirname, abspath
import os
import pandas as pd



# add to PythonPath the main root folder for correctly retrieve modules
root_dir = str(Path(__file__).absolute().parent.parent.parent)
sys.path.append(root_dir)



class MakeDataset:
    def __init__(self, conf, training=False):

        # Setting Conf
        self.conf = conf

        # Setting logging
        log_fmt = self.conf["GENERAL"]["logger_format"]
        logging.basicConfig(
            filename="MD.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(MakeDataset.__name__)

        self.folder_path      = Path(__file__).parent
        self.processed_folder = str(Path(__file__).absolute().parent.parent.parent / 'data/processed/')


    def main(self):
        """
        This function is responsible for retrieving all the data sources 
        """

        # START SIGNAL
        self.logger.info(" # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        self.logger.info("")
        self.logger.info("                                MPM START: make_embeddings")
        self.logger.info("")
        self.logger.info("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")

    