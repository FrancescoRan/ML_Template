import logging
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests 
import pandas as pd
import numpy as np
from typing import Union


class MultipleTests():
    
    def __init__(self, 
                 df : pd.DataFrame, 
                 target : str, 
                 test, 
                 alpha : float,
                 alternative : str = "two-sided", 
                 multiple_test_correction : bool = True):
        
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            filename="MD.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(MultipleTests.__name__)


        self.df          = df.select_dtypes(exclude=['object'])
        self.target      = target
        self.test        = test
        self.alternative = alternative
        self.alpha       = alpha
        self.multiple_test_correction = multiple_test_correction
    
    
    def main(self):
        
        self.logger.info('RUNNING MULTIPLE {} TEST'.format(self.test.__name__))

        res = self.compute_multiple_tests()
        
        return res


    def _test(self, 
              x : Union[np.array, pd.Series], 
              y : Union[np.array, pd.Series]) -> dict:

        test_res    = self.test(x = x, y = y, alternative = self.alternative)
        test_pvalue = np.round(test_res.pvalue, 12)

        res = {'vars': x.name, 'pval': test_pvalue}

        return res
    
    
    def compute_multiple_tests(self, 
                               control_value : list  = [], 
                               case_value : list = []) -> pd.DataFrame:

        if not control_value and not case_value:

            control_value = np.min(np.unique(self.df[self.target]))
            case_value    = np.max(np.unique(self.df[self.target]))
        
        controls = list(self.df[self.df[self.target] == control_value].index)
        cases    = list(self.df[self.df[self.target] == case_value].index)
        
        list_res_test = []
        
        for col in self.df.drop(self.target, axis = 1):
            
            try:
                
                res_test = self._test(x = self.df[col][controls], y = self.df[col][cases])
            
            except ValueError as e:
                
                print("Skipped var {} due to ValueError: {}".format(col, e))
                pass
            
            list_res_test.append(res_test)
        
        df_res = pd.DataFrame(list_res_test).sort_values("pval", ascending = True)
        
        # df_res = pd.DataFrame([self._test(x = self.df[col][controls], y = self.df[col][cases]) for col in self.df.drop(self.group_var, axis = 1)]).sort_values("pval", ascending = True)
        
        
        if self.multiple_test_correction:
            
            df_res['corrected_pvals'] = multipletests(pvals=df_res['pval'], 
                                                      alpha=self.alpha, 
                                                      method='fdr_bh', 
                                                      is_sorted=True)[1]
        
        df_res['n']    = len(controls + cases)
        df_res['frac'] = np.round(len(cases)/len(controls + cases), 2)
                    
        return df_res

    