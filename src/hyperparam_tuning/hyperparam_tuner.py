import logging
from pathlib import Path
from typing import Union
from numpy import min as np_min
from numpy import prod as np_prod

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score, accuracy_score


class HyperparamTuner:
    def __init__(self, conf):

        self.conf = conf

        # Setting logging
        log_fmt = self.conf["GENERAL"]["logger_format"]
        logging.basicConfig(
            filename="PB.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(HyperparamTuner.__name__)

        self.folder_path = Path(__file__).parent

        self.allowed_search_methods = ["grid", "random"]

        self.param_grid_model = self.conf['HYPERPARAM_TUNING']['param_grid_model']
        self.param_grid_embedder = self.conf['HYPERPARAM_TUNING']['param_grid_embedder']

    def scorer(self, scoring_functions: Union[str, list]) -> dict:

        self.logger.info(
            "The Model will be scored on {}".format(str(scoring_functions))
        )

        scorer_ = {}

        if not isinstance(scoring_functions, list):

            scoring_functions = [scoring_functions]

        if "f1" in scoring_functions:

            scorer_.update({"f1": make_scorer(fbeta_score, beta=1)})

        if "f2" in scoring_functions:

            scorer_.update({"f2": make_scorer(fbeta_score, beta=2)})

        if "roc_auc" in scoring_functions:

            scorer_.update({"roc_auc": make_scorer(roc_auc_score)})

        if "accuracy" in scoring_functions:

            scorer_.update({"accuracy": make_scorer(accuracy_score)})

        return scorer_

    def hyperparam_tuning(
        self,
        seed: int,
        search_type: str,
        # param_grid: dict,
        scoring_functions: Union[str, list],
        refit_metric: str,
        pipe: Pipeline,
        k_fold_splits: int,
        n_iter: int = 20,
    ):

        assert (
            search_type in self.allowed_search_methods
        ), "Please select a valid search_type in {allowed}, provided {search_type}".format(
            allowed=self.allowed_search_methods, search_type=search_type
        )

        param_grid = {}
        
        if pipe.named_steps['embedder']:
            embedder_name = pipe.named_steps['embedder'].transformers[0][0]
            param_grid.update({'embedder__{}__'.format(embedder_name) + k: v for k, v in self.param_grid_embedder.items()})

        
        if pipe.named_steps['model']:
            param_grid.update({"model__" + k: v for k, v in self.param_grid_model.items()})

        scoring = self.scorer(scoring_functions)

        skf = StratifiedKFold(n_splits=k_fold_splits, shuffle=True, random_state=seed)

        if search_type == "grid":

            self.logger.info(
                "RUNNING: GridSearchCV on {} K-fold CV".format(str(k_fold_splits))
            )

            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=scoring,
                cv=skf,
                refit=refit_metric,
                n_jobs=-1,
                verbose=3,
            )

        elif search_type == "random":

            self.logger.info(
                "RUNNING: RandomizedSearchCV on {} K-fold CV".format(str(k_fold_splits))
            )
            
            n_comb_param_grid = np_prod([len(v) for v in param_grid.values()])
            n_iter            = np_min([n_iter, n_comb_param_grid])
            
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=skf,
                refit=refit_metric,
                n_jobs=-1,
                verbose=3,
            )
        else:
            pass

        # search.fit(X, y)
        # print("Best parameter (CV score={}):".format(np.round(search.best_score_, 3)))
        # print(search.best_params_)

        return search
