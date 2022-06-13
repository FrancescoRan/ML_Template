import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)

from pathlib import Path
from typing import Union
import sys
from joblib import load
import os
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.inspection import permutation_importance
import shap


root_dir = str(Path(__file__).absolute().parent.parent.parent)
sys.path.append(root_dir)

from utils.utils import Utils


class FeaturesImportance:
    def __init__(self, conf):

        self.conf = conf

        # Setting logging
        log_fmt = self.conf["GENERAL"]["logger_format"]
        logging.basicConfig(
            filename="FI.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(FeaturesImportance.__name__)

        self.folder_path = Path(__file__).parent

        self.logger.info("STARTING: Features")

        """
        self.file_path = os.path.join(
            self.conf["MODELS"]["folder_to_save_model"],
            self.conf["MODELS"]["FEATURES_IMPORTANCE"]["model_name"],
        )
        """

    def main(self):

        pass

    def load_artifact(self, file_path):

        artifact = load(file_path)

        return artifact

    def get_features_importance(
        self,
        file_path_model: str,
        model_name: str,
        X_train: Union[pd.DataFrame, np.array],
        scoring: str,
        X_test: Union[pd.DataFrame, np.array, None] = None,
        y_test: Union[pd.Series, np.array, None] = None,
        type_: str = "impurity",
    ):
        assert type_ in [
            "impurity",
            "permutation",
            "shap",
        ], "Please select a valid type_ in ['impurity', 'permutation', 'shap']"

        
        file_path_model = os.path.join(file_path_model, model_name)
        print(file_path_model)
        artifact = self.load_artifact(file_path=file_path_model)
        
        try:
            model = artifact["model"]
        except TypeError:
            model = artifact.best_estimator_ # artifact.best_estimator_["model"]


        if type_ == "impurity":

            self.logger.info("RUNNING FEATURE IMPORTANCE: {type_}".format(type_=type_))

            feat_importance = np.round(model.feature_importances_, 3)

            df_feat_imp = pd.DataFrame(
                {
                    "features": X_train.columns,
                    "importance_impurity": feat_importance_mean,
                }
            ).sort_values(by="importance_impurity", ascending=False)

        elif type_ == "permutation":

            """
            assert (
                not X_test.empty()
            ), "Please provide a non empty X_test to compute permutation importance, otherwise select type_= 'impurity'"
            assert (
                not y_test.empty()
            ), "Please provide a non empy y_test to compute permutation importance, otherwise select type_= 'impurity'"
            """
            self.logger.info("RUNNING FEATURE IMPORTANCE: {type_}".format(type_=type_))

            feat_importance = permutation_importance(
                estimator=model,
                X=X_test,
                y=y_test,
                scoring=scoring,
                n_repeats=5,
                n_jobs=-1,
                max_samples=0.8,
            )
            feat_importance_mean = feat_importance.importances_mean
            feat_importance_std = feat_importance.importances_std

            df_feat_imp = (
                pd.DataFrame(
                    {
                        "features": X_train.columns,
                        "mean_importance_perm": feat_importance_mean,
                        "std_importance_perm": feat_importance_std,
                    }
                )
                .round(4)
                .sort_values(by="mean_importance_perm", ascending=False)
            )

        elif type_ == "shap":
            """
            assert (
                X_test
            ), "Please provide a non empty X_test to compute permutation importance, otherwise select type_= 'impurity'"
            assert (
                y_test
            ), "Please provide a non empy y_test to compute permutation importance, otherwise select type_= 'impurity'"
            """

            self.logger.info("RUNNING FEATURE IMPORTANCE: {type_}".format(type_=type_))

            X_train_sub = X_train.sample(n=X_train.shape[0], replace=False)
            X_test_sub = X_test.sample(n=X_test.shape[0], replace=False)
            explainer = shap.Explainer(model.predict_proba, X_train_sub, algorithm='permutation', max_evals=500)
            shap_values = explainer(X_test_sub)

            abs_avg_shap_values = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
            std_shap_values = shap_values.values[:, :, 1].std(axis=0)

            df_feat_imp = (
                pd.DataFrame(
                    {
                        "features": X_train.columns,
                        "mean_importance_shap": abs_avg_shap_values,
                        "std_importance_shap": std_shap_values,
                    }
                )
                .round(4)
                .sort_values(by="mean_importance_shap", ascending=False)
            )

            self.create_shap_plot(
                model_name=model_name,
                plot_type="bar",
                shap_values=shap_values,
                X_test_sub=X_test_sub,
                model=model,
            )

        return df_feat_imp

    def create_shap_plot(
        self,
        model_name: str,
        plot_type: str,
        shap_values,
        X_test_sub: pd.DataFrame
    ):

        # questa dovrebbe andare in uno script a parte: viz.py
        # feat_names = model.feature_names_in_
        plt.ioff()
        model_name = model_name.replace(".joblib", "")

        plt.figure(figsize=(16, 8))
        shap.summary_plot(
            shap_values.values[:, :, 1], X_test_sub, plot_type=plot_type
        )  # violin, bar, dot

        plots_folder = os.path.join(str(self.folder_path.parent), "plots")
        Utils.create_folder_if_not_exists(folder_path=plots_folder)
        
        plt.savefig(
            os.path.join(
                plots_folder,
                "shap_{plot_type}_{model}.png".format(
                    plot_type=plot_type, model=model_name
                ),
            ),
            dpi=600,
            bbox_inches="tight",
        )


if __name__ == "__main__":

    root_path = Path(__file__).absolute().parent.parent.parent
    config_path = root_path / "conf" / "conf.yml"

    conf = Utils.read_yaml(config_path)

    log_fmt = conf["GENERAL"]["logger_format"]
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info("MD - Start")

    FI = FeaturesImportance(conf)
    FI.main()

    logger.info(
        "MM - End --- Run time %s minutes ---" % ((time.time() - start_time) / 60)
    )
