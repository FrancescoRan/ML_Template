import logging

from sklearn.decomposition import PCA, FastICA, LatentDirichletAllocation, NMF
from sklearn.compose import ColumnTransformer, make_column_transformer

class Embedder:
    
    def __init__(self, conf):

        # Setting Conf
        self.conf = conf

        # Setting logging
        log_fmt = self.conf["GENERAL"]["logger_format"]
        logging.basicConfig(
            filename="EMB.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(Embedder.__name__)

        # self.folder_path      = Path(__file__).parent
        # self.processed_folder = os.path.join(str(Path(__file__).absolute().parent.parent.parent), 'data/processed/')
        
        self.method_name  = conf["EMBEDDINGS"]["EMBEDDER"]["method_name"]
        

        self.allowed_models = [
            None, 'pca', 'lda', 'fastica', 'nmf','autoencoder', 'var_autoencoder', 'custom'
        ]

    
    def main(self):
        """
        This function is responsible for retrieving all the data sources 
        """

        # START SIGNAL
        self.logger.info(" # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        self.logger.info("")
        self.logger.info("                                MPM START: Embedder")
        self.logger.info("")
        self.logger.info("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")


    def loading_method(self):

        assert (
            self.method_name in self.allowed_models
        ), "Please select a valid method, provided method_name: {}".format(
            self.method_name
        )

        if self.method_name == None:
            method = None

        if self.method_name == "pca":
            method = PCA()
        
        if self.method_name == "fastica":
            method = FastICA()
        
        if self.method_name == "lda":
            method = LatentDirichletAllocation()
        
        if self.method_name == "nmf":
            method = NMF()
        
        if self.method_name == "autoencoder":
            pass
        
        if self.method_name == "var_autoencoder":
            pass

        if self.method_name == "custom":
            pass

        return method


    def embedder(self, vars_to_embed: list):
        
        embedder = self.loading_method()
        """
        embedded_vars = ColumnTransformer(
            transformers=[("emb", embedder, vars_to_embed)],
            remainder = "passthrough"
        )
        """
        embedded_vars = make_column_transformer((embedder, vars_to_embed), remainder = "passthrough")
        return embedded_vars