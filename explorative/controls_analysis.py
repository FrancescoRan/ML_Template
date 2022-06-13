import pandas as pd
import numpy as np
import os
import seaborn as sns
# from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests 

from run_pca import run_PCA


class MultipleTests():
    
    def __init__(self, 
                 df : pd.DataFrame, 
                 target : str, 
                 test, alpha, multiple_test_correction = True):
        
        self.df          = df.select_dtypes(exclude=['object'])
        self.target      = target
        self.test        = test
        self.alternative = "two-sided"
        self.alpha       = alpha
        self.multiple_test_correction = multiple_test_correction
        
    
    def run(self):
        
        res = self.compute_multiple_tests()
        
        return res
    
    
    def _test(self, x, y):

        test_res    = self.test(x = x, y = y, alternative = self.alternative)
        test_pvalue = np.round(test_res.pvalue, 12)

        res = {'vars': x.name, 'pval': test_pvalue}

        return res
    
    
    def compute_multiple_tests(self, 
                               control_value : list  = [], 
                               case_value : list = []):

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
            
            df_res['corrected_pvals'] = multipletests(pvals = df_res['pval'], 
                                                      alpha=self.alpha, 
                                                      method='fdr_bh', 
                                                      is_sorted = True)[1]
        
        df_res['n'] = len(controls + cases)
        df_res['frac'] = np.round(len(cases)/len(controls + cases), 2)
                    
        return df_res
    

def run_MT_grouped(df, grouping_var, q):
    
    grouping_new_colname = '{}_q_{}'.format(grouping_var, str(q))
    df[grouping_new_colname] = pd.qcut(df[grouping_var], q)
    
    df = df.drop(grouping_var, axis = 1)
    
    qcut_levels = np.unique(df[grouping_new_colname]) 
    list_mw_res = []
    for lev in qcut_levels:

        df_test = df[df[grouping_new_colname] == lev]

        mt = MultipleTests(df = df_test, group_var = "outcome", test = mannwhitneyu, alpha = 0.05, multiple_test_correction = True)
        mw_res = mt.run()

        mw_res['qcut_level'] = lev

        list_mw_res.append(mw_res)

    multiple_test_by_q = pd.concat(list_mw_res).sort_values('corrected_pvals', ascending = True)
    
    return multiple_test_by_q



def _test(list_df_filenames):
    
    q_res         = []
    loadings_list = []
    
    for df_name in list_df_filenames:
        
        df         = pd.read_csv('/home/jupyter/vari/mpm/'+ df_name)
        cpg_vars   = [i for i in list(df.columns) if i[0:2] == "cg"]

        pca_cpg, df_cpg = run_PCA(df = df, 
                                  vars_to_consider = cpg_vars, 
                                  prefix_to_pc_cols = "cpg", 
                                  return_factor_scores = True,
                                  return_dataframe = True,
                                  threshold_var_to_retain = 0.005)
        
        pca_cpg_loadings = pd.DataFrame(pca_cpg.components_, columns = cpg_vars, index = ['cpg'+str(i) for i in range(pca_cpg.components_.shape[0])])

        
        df_mpm_pca   = pd.concat([df.drop(cpg_vars, axis = 1), df_cpg], axis = 1)
        vars_to_test = [c for c in df_mpm_pca.columns if "cpg" in c] + ['asbestos_exposure', 'outcome']
        df_mpm_pca   = df_mpm_pca[vars_to_test]

        q_        = run_MT_grouped(df = df_mpm_pca, grouping_var = 'asbestos_exposure', q = 5)
        q_['df_'] = df_name
        
        q_res.append(q_)
        loadings_list.append(pca_cpg_loadings)
        
    return q_res, loadings_list





######################################################################

df_mpm_pca = pd.read_csv('/home/jupyter/vari/mpm/df_tot_pca_0005.csv')

df = pd.read_csv('/home/jupyter/vari/mpm/df_cpg000001.csv')

cpg_vars   = [i for i in list(df.columns) if i[0:2] == "cg"]

pca_cpg, df_cpg = run_PCA(df = df, 
                          vars_to_consider = cpg_vars, 
                          prefix_to_pc_cols = "cpg", 
                          return_factor_scores = True,
                          return_dataframe = True,
                          threshold_var_to_retain = 0.005)

pca_cpg_loadings = pd.DataFrame(pca_cpg.components_, columns = cpg_vars, index = ['cpg'+str(i) for i in range(pca_cpg.components_.shape[0])])

pc_name = "cpg1"
np.abs(pca_cpg_loadings.T[pc_name].round(3)).sort_values(ascending = False)

# plot loading distr:
_comp = np.abs(pca_cpg_loadings.T[pc_name]).sort_values(ascending=False).reset_index()
plt.hist(_comp['cpg1'], bins = 100)


df_mpm_pca = pd.concat([df.drop(cpg_vars, axis = 1), df_cpg], axis = 1)
df_mpm_pca.shape

# df_mpm_pca = df_mpm_pca.drop([c for c in df_mpm_pca.columns if "ch" in c], axis = 1)
vars_to_test = [c for c in df_mpm_pca.columns if "cpg" in c] + ['asbestos_exposure', 'outcome']
df_mpm_pca = df_mpm_pca[vars_to_test]

q_0005 = run_MT_grouped(df = df_mpm_pca, grouping_var = 'asbestos_exposure', q = 5)
print(q_.head(10).to_string())
print(q_0005.head(10).to_string())


list_df_filenames = ['df_cpg001.csv', 'df_cpg0001.csv', 'df_cpg00001.csv', 'df_cpg000001.csv']
# list_df_filenames = ['df_cpg0001.csv', 'df_cpg00001.csv', 'df_cpg000001.csv']


q_res2, loadings_res2 = _test(list_df_filenames)

q_res        = q_res2.copy()
loadings_res = loadings_res2.copy()

counter = 0
pc_name = "cpg0"
for q_, load in zip(q_res, loadings_res):
    
    _comp_temp = np.abs(load.T[pc_name]).sort_values(ascending=False).reset_index()
    _comp_temp['rank'] = _comp_temp.index
    
    if counter == 0:
        _comp = _comp_temp
    else:
        _comp = _comp.merge(_comp_temp, how = "outer", on = 'index')
    
    counter =+ 1
    
    print(q_.head(10).to_string())
    print(_comp_temp)


_comp
[l.shape for l in loadings_res]




    

df_mpm_pca_shap_high_exp = df_mpm_pca[df_mpm_pca['asbestos_exposure'] > 1.5]
df_mpm_pca_shap_high_exp_contr = df_mpm_pca_shap_high_exp[df_mpm_pca_shap_high_exp['outcome'] == 0]
df_mpm_pca_shap_high_exp_cases = df_mpm_pca_shap_high_exp[df_mpm_pca_shap_high_exp['outcome'] == 1]

