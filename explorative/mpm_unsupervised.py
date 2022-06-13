import pandas as pd
import numpy as np
import os
import seaborn as sns
# from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests 


def _check_len_nans(df, axis, threshold = 0.3, plot_hist = False):
    
    assert threshold >= 0 and threshold <= 1, "threshold must be in [0,1]"
    
    df_na = df.isna().sum(axis)/df.shape[0]
    list_na_vars = list(df_na[df_na >= threshold].index)
    
    if plot_hist:
        plt.hist(df_na, log = True, bins = 60)
        plt.show()
    
    print(df_na.describe())
    
    df_na_cat = np.select(condlist = [
                            df_na == 0.0,   
                            df_na <= 0.0001,
                               df_na <= 0.005,
                               df_na <= 0.01,
                               df_na <= 0.05,
                               df_na <= 0.10,
                               df_na <= 0.2,
                               df_na > 0.2], 
                          choicelist = ['No Missing',
                                        '<= 0.0001',
                                        '<= 0.005%',
                                        '<= 1%',
                                        '<= 5%',
                                        '<= 10%',           
                                        '<= 20%',
                                        '> 20%']
                         )
    
    print(pd.Series(df_na_cat).value_counts())
    
    return df_na[df_na >= threshold]



def _impute_mean(df):
    ## su betas la media performa praticamente uguale alla soluzione migliore:
    ## https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03592-5
    
    df_na = df.isna().sum()
    list_na_vars = list(df_na[df_na > 0].index) 
    df[list_na_vars] = df[list_na_vars].apply(lambda x: x.fillna(x.mean(skipna = True)), axis = 0)
    
    return df



class MultipleTests():
    
    def __init__(self, df, group_var, test, alpha, multiple_test_correction = True):
        
        self.df          = df.select_dtypes(exclude=['object'])
        self.group_var   = group_var
        self.test        = test
        self.alternative = "two-sided"
        self.alpha       = alpha
        self.multiple_test_correction = multiple_test_correction
        
    
    def run(self):
        
        res = self.compute_multiple_tests()
        
        return res
    
    
    def _test(self, x, y):

        test_res    = self.test(x = x, y = y, alternative = self.alternative)
        test_pvalue = np.round(test_res.pvalue, 6)

        res = {'cpg': x.name, 'pval': test_pvalue}

        return res
    
    
    def compute_multiple_tests(self, control_value = [], case_value = []):

        if not control_value and not case_value:

            control_value = np.min(np.unique(self.df[self.group_var]))
            case_value    = np.max(np.unique(self.df[self.group_var]))
        
        controls = list(self.df[self.df[self.group_var] == control_value].index)
        cases    = list(self.df[self.df[self.group_var] == case_value].index)
        
        df_res = pd.DataFrame([self._test(x = self.df[col][controls], y = self.df[col][cases]) for col in self.df.drop(self.group_var, axis = 1)]).sort_values("pval", ascending = True)
        
        
        
        if self.multiple_test_correction:
            
            df_res['corrected_pvals'] = multipletests(pvals = df_res['pval'], 
                                                      alpha=self.alpha, 
                                                      method='fdr_bh', 
                                                      is_sorted = True)[1]
                    
        return df_res
    
    
    
    
    
    


    




def parallelize_dataframe(df, func, axis = 1, n_cores=os.cpu_count() - 1):
    
    import os
    
    if axis == 0:
        df_split = np.array_split(df, n_cores, axis = 0)
    else:
        df_split = np.array_split(df, n_cores, axis = 1)
        
    pool     = Pool(n_cores)
    df       = pd.concat(pool.map(func, df_split)) # .reset_index(drop = True)
    pool.close()
    pool.join()
        
    return df







## betas:

file_name_betas  = "MPM_case_control_betas_T.csv"
path_cwd  = os.getcwd()
file_path = os.path.join(path_cwd, file_name_betas)
# mpm_betas = pd.read_csv(file_path)

# mpm_betas.to_parquet('/home/jupyter/vari/mpm/betas/MPM_case_control_betas.parquet', index = False)

mpm_betas = pd.read_parquet('/home/jupyter/vari/mpm/betas/MPM_case_control_betas.parquet')
# mpm_betas = mpm_betas2.copy()

cpg_vars   = [i for i in list(mpm_betas.columns) if i[0:2] == "cg"]
ch_vars    = [i for i in list(mpm_betas.columns) if i[0:2] == "ch"]


mpm_betas_noch = mpm_betas.drop(ch_vars, axis = 1)

n_nans            = _check_len_nans(mpm_betas_noch, axis = 0, threshold = 0.2, plot_hist = True)
mpm_betas_reduced = mpm_betas_noch.drop(list(n_nans.index), axis = 1)
mpm_betas_reduced.shape


mpm_betas_reduced_imp = _impute_mean(mpm_betas_reduced)


#plt.hist(np.log(mpm_betas_reduced_imp['cg00000165'] + 0.00001), bins = 100)
#plt.hist(mpm_betas_reduced_imp['cg00000165'], bins = 100)





path_cwd  = os.getcwd() 
name_file = "MPM_case_control_phenotypes.csv"
path_import = os.path.join(path_cwd, name_file)

df_mpm = pd.read_csv(path_import, sep = ";")
 # 5 n ain smoke
df_mpm.loc[df_mpm['smoke'].isna(),'smoke'] =0


selected_vars = ['sampleID',
                'age',
                 'outcome',
                 'gender',
                 'asbestos_exposure',
                 'SNP_PC1',
                 'SNP_PC2',
                 'contr.probes.PC1',
                 'contr.probes.PC2',
                 'contr.probes.PC3',
                 'contr.probes.PC4',
                 'contr.probes.PC5',
                 'contr.probes.PC6',
                 'contr.probes.PC7',
                 'contr.probes.PC8',
                 'contr.probes.PC9',
                 'contr.probes.PC10',
                 'WBC_Horvath_PC1',
                 'BioAge4HAStaticAdjAge',
                 'AAHOAdjCellCounts']

df_mpm_selected = df_mpm[selected_vars]

df_all = df_mpm[['sampleID', 'outcome']].merge(mpm_betas_reduced_imp, how = 'inner', left_on = 'sampleID', right_on = 'Unnamed: 0')


mt = MultipleTests(df = df_all, group_var = "outcome", test = mannwhitneyu, alpha = 0.05, multiple_test_correction = True)
mw_res = mt.run()
# mw_res.to_csv('/home/jupyter/vari/mpm/mann_whitney_test.csv')

mw_res = pd.read_csv('/home/jupyter/vari/mpm/mann_whitney_test.csv')

# plt.hist(mw_res[mw_res['corrected_pvals'] <= 0.01]['corrected_pvals'], bins = 100)

vars_cgps001  = mw_res[mw_res['corrected_pvals'] <= 0.01]['cpg'].tolist()
vars_cgps0001 = mw_res[mw_res['corrected_pvals'] <= 0.001]['cpg'].tolist()
vars_cgps00001 = mw_res[mw_res['corrected_pvals'] <= 0.0001]['cpg'].tolist()
vars_cgps000001 = mw_res[mw_res['corrected_pvals'] <= 0.00001]['cpg'].tolist()


df_cpg001 = df_mpm.merge(mpm_betas_reduced_imp[['Unnamed: 0']+vars_cgps001], how = 'inner', left_on = 'sampleID', right_on = 'Unnamed: 0')
df_cpg001.to_csv('/home/jupyter/vari/mpm/df_cpg001.csv', index = False)

df_cpg0001 = df_mpm.merge(mpm_betas_reduced_imp[['Unnamed: 0']+vars_cgps0001], how = 'inner', left_on = 'sampleID', right_on = 'Unnamed: 0')
df_cpg0001.to_csv('/home/jupyter/vari/mpm/df_cpg0001.csv', index = False)

df_cpg00001 = df_mpm.merge(mpm_betas_reduced_imp[['Unnamed: 0']+vars_cgps00001], how = 'inner', left_on = 'sampleID', right_on = 'Unnamed: 0')
df_cpg00001.to_csv('/home/jupyter/vari/mpm/df_cpg00001.csv', index = False)

df_cpg000001 = df_mpm.merge(mpm_betas_reduced_imp[['Unnamed: 0']+vars_cgps000001], how = 'inner', left_on = 'sampleID', right_on = 'Unnamed: 0')
df_cpg000001.to_csv('/home/jupyter/vari/mpm/df_cpg000001.csv', index = False)

#################################### end make dataset ########################################

#### make_features:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import umap



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
    

# using df_cpg0001

df_cpg0001 = pd.read_csv('/home/jupyter/vari/mpm/df_cpg0001.csv')
# df = df_cpg0001.copy()
# df = pd.read_csv('/home/jupyter/vari/mpm/df_cpg000001.csv')
df_mpm_pca = pd.read_csv('/home/jupyter/vari/mpm/df_tot_pca_0005.csv')
df_mpm_pca.shape
cpg_vars   = [i for i in list(df.columns) if i[0:2] == "cg"]

pca_cpg, df_cpg = run_PCA(df = df, 
                          vars_to_consider = cpg_vars, 
                          prefix_to_pc_cols = "cpg", 
                          return_factor_scores = True,
                          return_dataframe = True,
                          threshold_var_to_retain = 0.005)

pca_cpg_loadings = pd.DataFrame(pca_cpg.components_, columns = cpg_vars)

n_comp = 0
_comp = np.abs(pca_cpg_loadings.iloc[n_comp,:]).sort_values(ascending=False).reset_index()
plt.hist(_comp[n_comp], bins = 100)


df_mpm_pca = pd.concat([df.drop(cpg_vars, axis = 1), df_cpg], axis = 1)

var_to_plot =  [c for c in df_mpm_pca.columns if "cpg" in c] + ['outcome']

plot_pca_cgp = sns.pairplot(df_mpm_pca[var_to_plot], 
                     hue="outcome", 
                     #corner=True,  
                     markers=["s", "D"], 
                     diag_kind="hist")
plot_pca_cgp.map_lower(sns.kdeplot, levels=4, color=".2")

name_plt = "pca_cpg000001.png"
path_cwd = os.getcwd()
plot_pca_cgp.savefig(os.path.join(path_cwd, name_plt))



## umap:

cpg_vars = [c for c in df_mpm_pca.columns if "cpg" in c]

# %%time
# mapper = umap.UMAP(random_state = 42).fit_transform(df_cpg, y=df['outcome'])
mapper = umap.UMAP(random_state = 42, n_neighbors=4).fit_transform(df_mpm_pca[cpg_vars], y=df_mpm_pca['outcome'])

umap_emb = pd.DataFrame(mapper, columns = ['umap1', 'umap2'])
df_umap_emb = pd.concat([df_mpm_pca[['outcome', 'asbestos_exposure']], umap_emb], axis = 1)

fig, ax = plt.subplots(figsize=(15, 12))
sns.scatterplot(ax = ax, data=df_umap_emb, x="umap1", y="umap2", hue="asbestos_exposure", style="outcome", size = "asbestos_exposure", sizes=(20, 200))



# hierarchical clustering:

from hdbscan import HDBSCAN

selected_vars = ['age',
                 'outcome',
                 'gender',
                 'asbestos_exposure',
                 'SNP_PC1',
                 'SNP_PC2',
                 'contr.probes.PC1',
                 'contr.probes.PC2',
                 'contr.probes.PC3',
                 'contr.probes.PC4',
                 'contr.probes.PC5',
                 'contr.probes.PC6',
                 'contr.probes.PC7',
                 'contr.probes.PC8',
                 'contr.probes.PC9',
                 'contr.probes.PC10',
                 'WBC_Horvath_PC1',
                 'BioAge4HAStaticAdjAge',
                 'AAHOAdjCellCounts'] 
selected_vars += [c for c in df_mpm_pca.columns if "cpg" in c]


# hdbscan
pipeline_hdbscan = make_pipeline(StandardScaler(), HDBSCAN(min_cluster_size=2))
clusterer = pipeline_hdbscan.fit(df_mpm_pca[selected_vars])

clusterer_hdbscan = clusterer['hdbscan']
pal = sns.color_palette('deep', 8)
colors = [sns.desaturate(pal[col], sat) for col, sat in zip(clusterer_hdbscan.labels_,
                                                        clusterer_hdbscan.probabilities_)]
plot_kwds={'alpha':0.25, 's':60, 'linewidths':0}
plt.scatter(df_mpm_pca['cpg0'], df_mpm_pca['cpg9'], c=colors, **plot_kwds)



### model:

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV, RepeatedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import SCORERS, classification_report, confusion_matrix, fbeta_score, accuracy_score,roc_auc_score,make_scorer, precision_score, recall_score
from lightgbm import LGBMClassifier, LGBMRegressor

from imblearn.over_sampling import BorderlineSMOTE

import shap


def get_cv_scores_from_pipeline(pipeline, X, y, cv_method, scoring_metrics):
    
    scores         = cross_validate(pipeline, X, y, cv = cv_method, scoring=scoring_metrics)
    scores_out     = {k: v for k, v in scores.items() if k != "fit_time" and k != "score_time"}
    scores_avg_out = {k: v.mean().round(4) for k, v in scores_out.items()}
    
    print("AVG SCORES:\n", {k: v for k, v in scores_avg_out.items()})
    
    return scores_out, scores_avg_out


selected_vars = ['age',
                 'outcome',
                 'gender',
                 'asbestos_exposure',
                 'SNP_PC1',
                 'SNP_PC2',
                 'contr.probes.PC1',
                 'contr.probes.PC2',
                 'contr.probes.PC3',
                 'contr.probes.PC4',
                 'contr.probes.PC5',
                 'contr.probes.PC6',
                 'contr.probes.PC7',
                 'contr.probes.PC8',
                 'contr.probes.PC9',
                 'contr.probes.PC10',
                 'WBC_Horvath_PC1',
                 'BioAge4HAStaticAdjAge',
                 'AAHOAdjCellCounts'] 
selected_vars += [c for c in df_mpm_pca.columns if "cpg" in c]



selected_vars2 = ['cpg0', 'cpg10', 'WBC_Horvath_PC1', 'cpg9', 'cpg11', 'SNP_PC1', 'cpg6', 
                'contr.probes.PC1', 'contr.probes.PC7', 'contr.probes.PC10', 'AAHOAdjCellCounts', 'cpg5', 'BioAge4HAStaticAdjAge', 'cpg13', 'contr.probes.PC3',
                'cpg12', 'cpg3', 'cpg1', 'contr.probes.PC5']


# target_var = "outcome"
sorted(SCORERS.keys())
f2_scorer = make_scorer(fbeta_score, beta=2)
f1_scorer = make_scorer(fbeta_score, beta=1)
f15_scorer = make_scorer(fbeta_score, beta=1.5)
accuracy_scorer = make_scorer(accuracy_score)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)
roc_auc_scorer = make_scorer(roc_auc_score)


# scoring = ['accuracy','f1','precision', 'recall', 'roc_auc', f2_scorer]
scoring = {'accuracy': accuracy_scorer,'f1': f1_scorer,'precision':precision_scorer, 'recall':recall_scorer, 'roc_auc':roc_auc_scorer, 'f2': f2_scorer, 'f15': f15_scorer}
target_var = "outcome"
vars_to_drop = [target_var] + ['asbestos_exposure']

df_X = df_mpm_pca[selected_vars].drop(vars_to_drop, axis =1)
# df_X = df_mpm_pca_0001[['cpg0', 'cpg14']]
df_y = df_mpm_pca[target_var]

x_train, x_test, y_train, y_test = train_test_split(df_X[selected_vars2], df_y, test_size = 0.30, stratify = df_y)

oversample = BorderlineSMOTE(sampling_strategy = "all", n_jobs = -1)
x_train_over, y_train_over = oversample.fit_resample(x_train, y_train)

rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=43)

# pipeline_lgb = make_pipeline(StandardScaler(), LGBMClassifier(num_leaves = 20,n_estimators=300, max_depth= 8, learning_rate=0.001))
# scores_lgb, _ = get_cv_scores_from_pipeline(pipeline = pipeline_lgb, X = df_X, y = df_y, cv_method = rskf, scoring_metrics = scoring)



pipeline_lgb = Pipeline(steps = [("standardscaler", StandardScaler()), ("lgb_C", LGBMClassifier())])


param_grid_lgb = {
    "lgb_C__num_leaves" : list(range(10, 20, 5)),
    "lgb_C__n_estimators" : list(range(50, 500, 100)),
    "lgb_C__max_depth" : list(range(2, 6, 2)),
    "lgb_C__learning_rate" : [0.1, 0.01, 0.001],
    "lgb_C__subsample" : [0.8]
}

lgb_best = GridSearchCV(estimator = pipeline_lgb, 
                        param_grid = param_grid_lgb, 
                        scoring = scoring,
                        n_jobs = -1, 
                        cv = rskf, 
                        refit = 'roc_auc',
                       verbose = 20)

lgb_best.fit(x_train_over, y_train_over)

lgb_best.cv_results_["mean_test_roc_auc"].max() # 0.84453
lgb_best.cv_results_["mean_test_f1"].max()
lgb_best.cv_results_["mean_test_recall"].max()

lgb    = lgb_best.best_estimator_['lgb_C']
# pred on test:
y_pred = lgb.predict(x_test)
print(classification_report(y_test, y_pred))
pd.DataFrame(confusion_matrix(y_test, y_pred), columns = ['pred_0', 'pred_1'], index = ['obs_0', 'obs_1'])


id_best_hyperparams = [ix for ix, i in enumerate(lgb_best.cv_results_["rank_test_roc_auc"]) if i == 1]
np.array([lgb_best.cv_results_[k] for k in lgb_best.cv_results_.keys() if 'params' in k][0])[id_best_hyperparams]




### shap


def model_proba(x):
    return lgb.predict_proba(x)[:,1]

model = lgb
training_features = x_train_over.columns
background_ = shap.maskers.Independent(df_X[training_features], max_samples=df_X.shape[0])
explainer   = shap.Explainer(model_proba, background_)
# shap_values2 = explainer(df_X[training_features])
shap_values = shap.TreeExplainer(model).shap_values(df_X)
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(df_X)

shap.plots.bar(shap_values2[1:2].cohorts(2).abs.mean(0))
shap.plots.beeswarm(shap_values, max_display = 15)
shap.plots.heatmap(shap_values, max_display = 7)

for i, feat in enumerate(training_features):
    shap.dependence_plot(i, shap_values[1], df_X)

shap.summary_plot(shap_interaction_values, df_X)



shap.dependence_plot(("age", "age"), shap_interaction_values, df_X)
shap.dependence_plot(("cpg4", "cpg0"), shap_interaction_values, df_X)

tmp = np.abs(shap_interaction_values).sum(0)
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
plt.figure(figsize=(12,12))
plt.imshow(tmp2)
plt.yticks(range(tmp2.shape[0]), df_X.columns[inds], rotation=50.4, horizontalalignment="right")
plt.xticks(range(tmp2.shape[0]), df_X.columns[inds], rotation=50.4, horizontalalignment="left")
plt.gca().xaxis.tick_top()
plt.show()








training_features = df_X.columns
shap_values = shap.TreeExplainer(lgb).shap_values(df_X[training_features], check_additivity=True)
shap.summary_plot(shap_values[0], df_X[training_features], plot_type = "bar")
shap.summary_plot(shap_values[1], df_X[training_features], max_display = 20)
# shap.dependence_plot('cpg0', shap_values[1], df_X[training_features], interaction_index="cpg13")
# shap.dependence_plot('SNP_PC1', shap_values[1], df_X, interaction_index="cpg1")
shap.plots.heatmap(shap_values[1])

### mettiamo insieme con asbestos:
df_shap_values = pd.DataFrame(shap_values.values, columns = ["shap_"+ f for f in training_features])
df_mpm_pca_shap = pd.concat([df_mpm_pca_0001, df_shap_values], axis = 1)
df_mpm_pca_shap[['asbestos_exposure', "cpg0"]].corr()
sns.jointplot(x=df_mpm_pca_shap['asbestos_exposure'], y=df_mpm_pca_shap['cpg0'], kind="hex", color="#4CB391")
# prova robust pca usa sia L che S

######## high exposure
df_mpm_pca_shap_high_exp = df_mpm_pca_shap[df_mpm_pca_shap['asbestos_exposure'] > 1.5]
df_mpm_pca_shap_high_exp_contr = df_mpm_pca_shap_high_exp[df_mpm_pca_shap_high_exp['outcome'] == 0]
df_mpm_pca_shap_high_exp_cases = df_mpm_pca_shap_high_exp[df_mpm_pca_shap_high_exp['outcome'] == 1]


def get_shap_values(df):
    
    background  = shap.maskers.Independent(df, max_samples=df.shape[0])
    explainer   = shap.Explainer(model_proba, background)
    shap_values = explainer(df)
    
    return shap_values


shap_values_high_contr = get_shap_values(df = df_mpm_pca_shap_high_exp_contr[training_features])

shap.summary_plot(shap_values_high_contr, max_display = 10)


shap.plots.waterfall(shap_values_high_contr[18], max_display=14)
shap.plots.heatmap(shap_values_high_contr)

df_mpm_pca_shap_high_exp_contr.iloc[16,:]['asbestos_exposure']

shap.summary_plot(shap_values[1], df_X[training_features], max_display = 10)



plot_cpg0_exposure = sns.pairplot([['asbestos_exposure', "shap_cpg0", 'outcome']], 
                     #hue="outcome", 
                     #corner=True,  
                     markers=["s", "D"], 
                     diag_kind="hist")
