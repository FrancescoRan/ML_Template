import logging

# from msilib.schema import Error
from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


# add to PythonPath the main root folder for correctly retrieve modules
root_dir = str(Path(__file__).absolute().parent.parent.parent)
sys.path.append(root_dir)


class Model:
    def __init__(self, conf):

        self.conf = conf

        # Setting logging
        log_fmt = self.conf["GENERAL"]["logger_format"]
        logging.basicConfig(
            filename="M.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(Model.__name__)

        self.folder_path = Path(__file__).parent

        self.model_name = conf["MODELS"]["model_name"]
        self.problem_type = conf["MODELS"]["problem_type"]

        self.allowed_models = [
            "rf",
            "logistic",
            "gbm",
            "lgbm",
            "xgb",
            "linear",
            "custom",
            'None'
        ]

    def loading_model(self):

        assert (
            self.model_name in self.allowed_models
        ), "Please select a valid model in {}, provided model_name: {}".format(
            self.allowed_models,
            self.model_name
        )

        if self.model_name == 'None':
            model = None # rompe tutto

        if self.model_name == "rf":
            if self.problem_type == "classification":
                model = RandomForestClassifier()

            else:
                model = RandomForestRegressor()

        if self.model_name == "gbm":
            if self.problem_type == "classification":
                model = GradientBoostingClassifier()

            else:
                model = GradientBoostingRegressor()

        if self.model_name == "lgbm":
            if self.problem_type == "classification":
                model = LGBMClassifier()

            else:
                model = LGBMRegressor()

        if self.model_name == "xgb":
            if self.problem_type == "classification":
                model = XGBClassifier(use_label_encoder=False, verbosity = 0)

            else:
                model = XGBRegressor(use_label_encoder=False, verbosity = 0)

        if self.model_name == "logistic":
            if self.problem_type == "classification":
                model = LogisticRegression()

            else:
                raise Exception(
                    "logistic is used for classification problems, provided problem_type: {}".format(
                        self.problem_type
                    )
                )

        if self.model_name == "linear":
            if self.problem_type == "regression":
                model = LinearRegression()

            else:
                raise Exception(
                    "linear is used for regression problems, provided problem_type: {}".format(
                        self.problem_type
                    )
                )

        if self.model_name == "custom":

            print("Provide Custom Model and add __str__() modlue")
            pass

        self.logger.info("Selecting {}".format(model.__str__()))

        return model

    def save_model(self, path, model, model_name):

        self.logger.info("Saving model {} to ".format(model.__str__()))

        pass
