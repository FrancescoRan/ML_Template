import logging
import time
from pathlib import Path
import sys
from os.path import dirname, abspath
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
from typing import Union
import os

# from scipy.stats import mannwhitneyu
import scipy.stats

# add to PythonPath the main root folder for correctly retrieve modules
root_dir = str(Path(__file__).absolute().parent.parent.parent)
sys.path.append(root_dir)

from utils.time_utils.timer_cm import Timer
from utils.utils import Utils as ut
from utils.clean_utils import CleanUtils as cu

from src.dataset.multiple_test import MultipleTests 

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

        self.mpm_betas_filename       = self.conf['DATA']['INTERIM']['mpm_betas_filename']
        self.mpm_betas_cols_to_select = self.conf['DATA']['INTERIM']['mpm_betas_cols_to_select']
        self.folder_name              = self.conf['DATA']['INTERIM']['folder_name']

        self.mpm_pheno_filename       = self.conf['DATA']['PROCESSED']['mpm_pheno_filename']
        self.folder_name_pheno        = self.conf['DATA']['PROCESSED']['folder_name']


        self.mpm_file_path       = os.path.join(str(self.folder_path.parent.parent / "data" ), self.folder_name, self.mpm_betas_filename)
        self.mpm_pheno_file_path = os.path.join(str(self.folder_path.parent.parent / "data" ), self.folder_name_pheno, self.mpm_pheno_filename)

        self.target              = self.conf['DATASET']['MULTIPLE_TEST']['target']
        self.test_name           = self.conf['DATASET']['MULTIPLE_TEST']['test_name']
        self.pval_thresholds     = self.conf['DATASET']['MULTIPLE_TEST']['pval_thresholds']
        
        self.test = getattr(scipy.stats, self.test_name)

    
    def main(self):
        """
        This function is responsible for retrieving all the data sources 
        """

        # START SIGNAL
        self.logger.info(" # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        self.logger.info("")
        self.logger.info("                                MPM START: make_dataset")
        self.logger.info("")
        self.logger.info("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")


        # df_mpm_betas = self.load_data(file_path=self.mpm_file_path, cols_to_select=self.mpm_betas_cols_to_select)
        df_mpm_betas = self.load_data(file_path=self.mpm_file_path)
        df_mpm_pheno = self.load_data(file_path=self.mpm_pheno_file_path, delimiter=';')
        
        df = df_mpm_pheno.merge(df_mpm_betas, how='left', left_on='sampleID', right_on='Unnamed:_0')
        df = cu._impute_mean(df=df)
        
        # _ = cu._check_len_nans(df=df_mpm_betas, axis=1, threshold=0.01, return_df=False)

        ## Multiple test:
        cg_colnames  = [c for c in df.columns if 'cg' in c]
        cols_to_test = [self.target] +  cg_colnames # test only cg
        
        mt          = MultipleTests(df=df[cols_to_test], 
                                    target=self.target, 
                                    test=self.test, 
                                    alpha=0.05, 
                                    multiple_test_correction=True)
        df_res_test = mt.main()

        for threshold in self.pval_thresholds:
            df_select = self.select_columns_from_pval(df=df, 
                                                      df_test_results=df_res_test, 
                                                      pval_colname='corrected_pvals', 
                                                      pval_threshold=threshold)
            
            if not df_select.empty:
                df_out = pd.concat([df.drop(cg_colnames, axis = 1), df_select], axis = 1).drop('Unnamed:_0', axis = 1)
                df_out = df_out.round(7)
                name_df_out = 'df_cg_pval_{}.csv'.format(str(threshold))
                
                print('Saving selected cg as {name_df_out} with shape: {shape_}'.format(name_df_out=name_df_out, shape_=str(df_out.shape)))
                df_out.to_csv(os.path.join(self.processed_folder, name_df_out), sep = ';')

        
        


    def load_data(self, 
                  file_path : str, 
                  cols_to_select : Union[list,str] = None,
                  delimiter: Union[str,None] = None) -> pd.DataFrame:

        file_name = file_path.split('/')[-1]
        self.logger.info('LOADING {}'.format(file_name))

        file_type = file_path.split('.')[-1]

        df = ut.read_file_from_memory(
                    file_path=file_path,
                    file_type=file_type,
                    columns_to_select=cols_to_select,
                    delimiter=delimiter
                )

        print("DF SHAPE:", df.shape)
        
        return df

    
    def select_columns_from_pval(self,
                                 df : pd.DataFrame,
                                 df_test_results : pd.DataFrame, 
                                 pval_colname : str,
                                 pval_threshold : float = 0.01):
        
        assert all(var in df_test_results.columns for var in ['vars', pval_colname]), "Please provide both 'vars' and {} in df".format(pval_colname)

        vars_to_select = df_test_results[df_test_results[pval_colname] <= float(pval_threshold)]['vars'].tolist()
        df_selected    = df[vars_to_select]

        return df_selected



if __name__ == "__main__":

    root_path = Path(__file__).absolute().parent.parent.parent
    config_path = root_path / "conf" / "conf.yml"

    conf = ut.read_yaml(config_path)

    log_fmt = conf["GENERAL"]["logger_format"]
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info("MD - Start")

    MD = MakeDataset(conf, training=True)
    MD.main()

    logger.info(
        "MD - End --- Run time %s minutes ---" % (
            (time.time() - start_time) / 60)
    )
