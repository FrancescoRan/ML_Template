import logging
import time
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import Union
from joblib import dump, load
import os
import re
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# add to PythonPath the main root folder for correctly retrieve modules
root_dir = str(Path(__file__).absolute().parent.parent.parent)
sys.path.append(root_dir)

from utils.utils import Utils
from utils.training_utils import _get_model_name, _set_name_to_artifact, dump_artifact

from src.hyperparam_tuning.hyperparam_tuner import HyperparamTuner
from src.pipeline.pipeline_builder import PipelineBuilder

from src.feature_importance.features_importance import FeaturesImportance

class TrainModel:
    def __init__(self, conf, training=True):

        self.conf = conf

        # Setting logging
        log_fmt = self.conf["GENERAL"]["logger_format"]
        logging.basicConfig(
            filename="TM.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(TrainModel.__name__)

        self.folder_path = Path(__file__).parent

        self.root_path = Path(__file__).parent.parent.parent.absolute()

        self.in_memory_folder_input = os.path.join(self.root_path, self.conf['TRAINING']['in_memory_folder_input'])
        self.artifacts_folder       = os.path.join(self.root_path, self.conf["TRAINING"]["artifacts_folder"])

        self.feature_imp_folder = os.path.join(
            str(self.folder_path.parent.parent),
            self.conf["FEATURES_IMPORTANCE"]["feature_importance_folder"],
        )
        self.cv_results_folder = os.path.join(
            str(self.folder_path.parent.parent), "data/cv_results",
        )

        self.select_features_subset = self.conf["TRAINING"]["select_features_subset"]
        self.vars_to_load           = self.conf["TRAINING"]["vars_to_load"]
        self.vars_to_drop_in_train  = self.conf["TRAINING"]["vars_to_drop_in_train"]
        self.name_input_tab         = self.conf["TRAINING"]["name_input_tab"]

        self.training      = training
        self.seed          = self.conf["TRAINING"]["seed"]
        self.target_var    = self.conf["TRAINING"]["target_var"]
        self.perc_train    = self.conf["TRAINING"]["perc_train"]
        self.scaler_type   = self.conf["TRAINING"]["scaler_type"]
        self.search_scaler = self.conf["TRAINING"]["search_scaler"]

        self.scoring_functions = self.conf["TRAINING"]["scoring_functions"]
        self.refit_metric      = self.conf["TRAINING"]["refit_metric"]
        self.search_type       = self.conf["TRAINING"]["search_type"]
        self.n_iter            = self.conf["TRAINING"]["n_iter"]
        self.k_fold_splits     = self.conf["TRAINING"]["k_fold_splits"]

        self.feat_imp_type      = self.conf["FEATURES_IMPORTANCE"]["feat_imp_type"]
        

        

    def main(self) -> None:

        """
        This function is responsible for retrieving all the data sources 
        """

        # START SIGNAL
        self.logger.info("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        self.logger.info("")
        self.logger.info("                                 MODEL: TRAINING")
        self.logger.info("")
        self.logger.info("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")

        ut = Utils()

        ut.create_folder_if_not_exists(self.artifacts_folder)
        ut.create_folder_if_not_exists(self.feature_imp_folder)
        ut.create_folder_if_not_exists(self.cv_results_folder)


        if self.select_features_subset:
            
            file_path = os.path.join(self.in_memory_folder_input, self.name_input_tab)
            colnames  = pd.read_csv(file_path, index_col=0, nrows=0, sep = ";").columns.tolist()
            
            sub_colnames = [c for c in colnames if 'cg' in c]

            df = ut.load_data(name_input_tab=self.name_input_tab, 
                              in_memory_folder_input=self.in_memory_folder_input, 
                              from_memory=True, 
                              columns_to_select=self.vars_to_load + sub_colnames)
        
        else:
            df = ut.load_data(from_memory=True)
        
        vars_to_embed    = [c for c in df.columns if 'cg' in c]
         
        # chr_vars_to_drop = [c for c in df.columns if 'ch' in c]
        
        
        # df = self._preprocess_data(df=df, target_var=self.target_var)

        X_train, X_test, y_train, y_test = self.split_train_test(
            df=df,
            perc_train=self.perc_train,
            target_var=self.target_var,
            seed=self.seed,
            random=True,
        )

        pb = PipelineBuilder(self.conf)

        pipeline, scaler_param = pb.pipeline_builder(
                vars_to_embed=vars_to_embed,
                encode_target_var=False,
                search_scaler=self.search_scaler, 
                scaler_type=self.scaler_type
            )

        ht = HyperparamTuner(self.conf)
        hyperparam_tuner = ht.hyperparam_tuning(
            seed=self.seed,
            search_type=self.search_type,
            # param_grid=self.param_grid,
            scoring_functions=self.scoring_functions,
            refit_metric=self.refit_metric,
            pipe=pipeline,
            k_fold_splits=self.k_fold_splits,
            n_iter=self.n_iter,
        )

        hyperparam_tuner.fit(X_train.drop(self.vars_to_drop_in_train, axis=1), y_train)

        model_name = _get_model_name(estimator=hyperparam_tuner, pipeline=True)

        # dump_artifact(pipeline_estimator=hyperparam_tuner, artifacts_folder=self.artifacts_folder, artifact_name=model_name)
    
        df_hyperparam_tuner_cv_res = self.test_evaluation(
            hyperparam_tuner=hyperparam_tuner, 
            vars_to_drop_in_train=self.vars_to_drop_in_train,
            X_test=X_test, 
            y_test=y_test
        )

        full_model_name, _ = self.save_model(
            hyperparam_tuner=hyperparam_tuner, 
            artifacts_folder=self.artifacts_folder,
            model_name=model_name
        )

        model_name_to_compute_feat_imp = np.where(
            self.conf["FEATURES_IMPORTANCE"]["model_name"] != "",
            self.conf["FEATURES_IMPORTANCE"]["model_name"],
            full_model_name,
        ).item()

        fi = FeaturesImportance(self.conf)

        feat_imp = fi.get_features_importance(
            file_path_model=self.artifacts_folder,
            model_name=model_name_to_compute_feat_imp,
            X_train=X_train.drop(self.vars_to_drop_in_train, axis=1),
            scoring=self.refit_metric,
            X_test=X_test.drop(self.vars_to_drop_in_train, axis=1),
            y_test=y_test,
            type_=self.feat_imp_type,
        )

        feat_imp_df_name = "df_imp_{model}_{type_}.csv".format(
            model=model_name_to_compute_feat_imp.split(".")[0],
            type_=self.feat_imp_type,
        )

        cv_res_df_name = "df_cv_res_{model}.csv".format(
            model=model_name_to_compute_feat_imp.split(".")[0]
        )

        feat_imp.to_csv(
            os.path.join(self.feature_imp_folder, feat_imp_df_name), index=False
        )

        df_hyperparam_tuner_cv_res.to_csv(
            os.path.join(self.cv_results_folder, cv_res_df_name), index=False
        )




    def split_train_test(
        self,
        df: pd.DataFrame,
        perc_train: float,
        target_var: str,
        seed: Union[int, float],
        random: bool = True,
    ) -> pd.DataFrame:

        """
        :param 
            df: pd.DataFrame
            perc_train: float | must be between 0 and 1
            target_name: str
            seed: int
            random: bool       
        :return: 
            pd.DataFrame
        """

        assert (
            isinstance(perc_train, float) and perc_train < 1 and perc_train > 0
        ), "Please select a valid perc_train: float >0 and <1"
        assert (
            target_var in df.columns
        ), "target name not in df.columns, provided {}".format(target_var)

        self.logger.info("SPLITTING TYPE: random == {}".format(random))
        self.logger.info("DF BEFORE SPLIT SHAPE: {}".format(str(df.shape)))

        if random == True:

            X_train, X_test, y_train, y_test = train_test_split(
                df,
                df[target_var],
                stratify=df[target_var],
                random_state=seed,
                shuffle=True,
            )

        else:

            threshold_train = int(df.shape[0] * perc_train)
            X_train = df.iloc[:threshold_train, :]
            X_test = df.iloc[threshold_train:, :]

            y_train = df[target_var].iloc[:threshold_train]
            y_test = df[target_var].iloc[threshold_train:]

        self.logger.info("DF TRAIN SHAPE: {}".format(str(X_train.shape)))
        self.logger.info("DF TEST SHAPE: {}".format(str(X_test.shape)))

        return X_train, X_test, y_train, y_test

    def test_evaluation(
        self,
        hyperparam_tuner,
        vars_to_drop_in_train : list,
        X_test: Union[pd.DataFrame, np.array],
        y_test: Union[pd.DataFrame, np.array],
    ) -> pd.DataFrame:

        print(
            "\nScoring {} on train: ".format(self.refit_metric),
            hyperparam_tuner.best_score_,
        )

        best_estimator = hyperparam_tuner.best_estimator_
        y_pred = best_estimator.predict(X_test.drop(vars_to_drop_in_train, axis=1))

        cm = pd.DataFrame(
            confusion_matrix(y_true=y_test, y_pred=y_pred),
            columns=["pred_" + str(l) for l in np.unique(y_test)],
            index=["obs_" + str(l) for l in np.unique(y_test)],
        )

        class_report = classification_report(y_true=y_test, y_pred=y_pred)
        score = hyperparam_tuner.score(
            X_test.drop(vars_to_drop_in_train, axis=1), y_test
        )

        hyperparam_tuner_cv_res = {
            k: v
            for k, v in hyperparam_tuner.cv_results_.items()
            for j in ["param_", "mean_test_"]
            if j in k
        }

        df_hyperparam_tuner_cv_res = (
            pd.DataFrame(hyperparam_tuner_cv_res)
            .sort_values(by="mean_test_{}".format(self.refit_metric), ascending=False)
            .round(3)
        )

        print("Scoring {} on test: ".format(self.refit_metric), score)
        print("Best Hyperparameters: {}".format(hyperparam_tuner.best_params_))
        print("Confusion Matrix:")
        print(cm)
        print("\n")
        print("Classification Report:")
        print(class_report)

        return df_hyperparam_tuner_cv_res


    def save_model(self, hyperparam_tuner, artifacts_folder, model_name: str) -> None:

        pipeline_estimator = hyperparam_tuner.best_estimator_

        self.logger.info("Saving pipeline: {}".format(pipeline_estimator))

        model_name = dump_artifact(pipeline_estimator=hyperparam_tuner, 
                                   artifacts_folder=artifacts_folder, 
                                   artifact_name=model_name)

        self.logger.info(
            "Model Saved as {model_name} in {artifacts_folder}".format(
                model_name=model_name, artifacts_folder=artifacts_folder
            )
        )

        return model_name, artifacts_folder


if __name__ == "__main__":

    root_path = Path(__file__).absolute().parent.parent.parent
    config_path = root_path / "conf" / "conf.yml"

    conf = Utils.read_yaml(config_path, arrays=True)

    log_fmt = conf["GENERAL"]["logger_format"]
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info("TM - Start")

    TM = TrainModel(conf, training=True)
    TM.main()

    logger.info(
        "TM - End --- Run time %s minutes ---" % ((time.time() - start_time) / 60)
    )
