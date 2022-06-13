import logging
from pydoc import classname
from typing import Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    
)


def identity_function(x):

    return x


def IdentityScaler():

    scaler = FunctionTransformer(identity_function)

    return scaler

class WrapperScaler(BaseEstimator, TransformerMixin):

    def __init__(self, scaler) -> None:
        self.scaler = scaler
        
    def fit(self, X, y = None):
        
        self.scaler.fit(X)
        return self

    def transform(self, X, y = None):
        
        feature_names = X.columns
        trans         = pd.DataFrame(self.scaler.transform(X), columns=feature_names)

        return trans

class Scaler():

    def __init__(self, conf) -> None:
        
        self.conf = conf

        # Setting logging
        log_fmt = self.conf["GENERAL"]["logger_format"]
        logging.basicConfig(
            filename="Sc.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(Scaler.__name__)

        self.logger.info("Initialized: Scaler")

    
    def features_scaler(
        self, 
        scaler_type: Union[str, list], 
        search_scaler: bool = False
    ) -> Union[list, FunctionTransformer, StandardScaler, Normalizer, RobustScaler, MinMaxScaler]:

        """
        :param
            search: bool 
            type_: str 
        :return: 
            list | FunctionTransformer | StandardScaler | Normalizer | RobustScaler | MinMaxScaler
        """

        allowed_scalers = ["identity", "standard", "minmax", "norm", "robust"]

        if search_scaler:

            scaler = []

            if not isinstance(scaler_type, list):

                scaler_type = [scaler_type]

            if "identity" in scaler_type:

                scaler.append(IdentityScaler())  # does nothing: no scaler

            if "standard" in scaler_type:

                scaler.append(StandardScaler())
            
            if "minmax" in scaler_type:

                scaler.append(MinMaxScaler())

            if "norm" in scaler_type:

                scaler.append(Normalizer())

            if "robust" in scaler_type:
                scaler.append(RobustScaler())

        else:

            assert (
                scaler_type in allowed_scalers
            ), "scaler_type (conf.yml) is not in {allowed}, provided {type_}".format(
                allowed=allowed_scalers, type_=scaler_type
            )

            if scaler_type == "identity":

                scaler = IdentityScaler()  # does nothing: no scaler

            if scaler_type == "standard":

                scaler = StandardScaler()
            
            if scaler_type == "minmax":

                scaler = MinMaxScaler()

            elif scaler_type == "norm":

                scaler = Normalizer()

            elif scaler_type == "robust":

                scaler = RobustScaler()

            else:
                # inserire una funzione custom
                pass

            self.logger.info("TRANSFORMING FEATURES: {}".format(scaler_type))

        wrapper_scaler = WrapperScaler(scaler)

        return wrapper_scaler

