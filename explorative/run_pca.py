import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os



def run_PCA(df,
            vars_to_consider, 
            prefix_to_pc_cols = None, 
            return_factor_scores = False, 
            return_dataframe = False, 
            threshold_var_to_retain = None,
            **kwargs
           ):
    
    assert return_dataframe and isinstance(prefix_to_pc_cols, str), "\nPlease provide a prefix_to_pc_cols : str or set return_dataframe to False"
    
    from sklearn.decomposition import PCA
    
    
    
    if kwargs:
        pca = PCA(kwargs)
    
    else:
        pca = PCA()
        pca.fit(df[vars_to_consider])
        

    if return_factor_scores:
        
        print("\nComputing Factor Scores")
        
        pc_to_retain = [i for i, v in enumerate(pca.explained_variance_ratio_) if v >= threshold_var_to_retain] 
        
        fs = pca.fit_transform(df[vars_to_consider])

        if threshold_var_to_retain is not None and return_dataframe:         
            assert len(pc_to_retain) > 0, "No PC has variance explaned ratio >= threshold_var_to_retain.\nPlease decrease threshold_var_to_retain to get some results"
            
            fs    = fs[np.ix_(np.arange(fs.shape[0]), pc_to_retain)]
            df_fs = pd.DataFrame(fs, columns = [prefix_to_pc_cols + str(i) for i in range(fs.shape[1])])
            
        elif threshold_var_to_retain is not None and return_dataframe == False:
            assert len(pc_to_retain) > 0, "No PC has variance explaned ratio >= threshold_var_to_retain.\nPlease decrease threshold_var_to_retain to get some results"
             
            df_fs = fs[np.ix_(np.arange(fs.shape[0]), pc_to_retain)]
            
        elif threshold_var_to_retain is None and return_dataframe:
                
            df_fs = pd.DataFrame(fs, columns = [prefix_to_pc_cols + str(i) for i in range(fs.shape[1])])

        
        print("\nPCA Completed!", "\nFactor Scores shape:", df_fs.shape)
        print("\nExplained Variance Ratio\n", np.round(pca.explained_variance_ratio_[:df_fs.shape[1]], 4))
        print("\nCumulated Variance\n", np.cumsum(np.round(pca.explained_variance_ratio_[:df_fs.shape[1]], 4)))
        
        return pca, df_fs

    else:
        
        return pca
    