import logging
from pathlib import Path
import sys
from typing import Union

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# add to PythonPath the main root folder for correctly retrieve modules
root_dir = str(Path(__file__).absolute().parent.parent.parent)
sys.path.append(root_dir)

from src.models.model import Model
from src.scalers.scaler import Scaler
from src.embedding.embedder import Embedder


class PipelineBuilder:
    def __init__(self, conf):

        self.conf = conf

        # Setting logging
        log_fmt = self.conf["GENERAL"]["logger_format"]
        logging.basicConfig(
            filename="PB.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(PipelineBuilder.__name__)

        self.folder_path = Path(__file__).parent

        self.scaler_type   = self.conf['SCALERS']['scaler_type']
        self.search_scaler = self.conf['SCALERS']['search_scaler']

        self.scaler = Scaler(self.conf)

        
        
    
    def target_encoder(self, target_var: str):

        target_enc = ColumnTransformer(
            transformers=[("target_var", LabelEncoder, [target_var])]
        )

        return target_enc
    
    
    def load_model(self):

        m = Model(self.conf)
        model = m.loading_model()

        return model


    def load_embedder(self):

        e = Embedder(self.conf)
        embedder = e.loading_method()

        return embedder



    def pipeline_builder(
        self,
        vars_to_embed : list,
        encode_target_var: Union[str, bool] = False,
        search_scaler: bool = False, 
        scaler_type: Union[str, list] = 'standard'
    ) -> Union[Pipeline and None, Pipeline and dict]:


        self.logger.info("BUILDING THE TRAINING PIPELINE")
        print("Setting search_scaler == {}".format(search_scaler))

        if encode_target_var:
            target_encoder = target_encoder(target_var=encode_target_var)
        else:
            target_encoder = None

        model    = self.load_model()

        e        = Embedder(self.conf)
        embedder = e.embedder(vars_to_embed=vars_to_embed)
        
        if search_scaler:

            scaler_param = {
                "scaler__": self.scaler.features_scaler()(
                    type_=scaler_type, search_scaler=search_scaler
                )
            }
            pipe = Pipeline(
                steps=[('embedder', embedder), ('target_encoder', target_encoder), ("model", model)]
            )  # capire la presenza di scaler qui che deve essere selezionato come hyperparam

            return pipe, scaler_param

        else:

            scaler = self.scaler.features_scaler(
                scaler_type=scaler_type, search_scaler=search_scaler
            )

            pipe = Pipeline(steps=[("scaler", scaler), ('embedder', embedder), ('target_encoder', target_encoder), ("model", model)])

        return pipe, None