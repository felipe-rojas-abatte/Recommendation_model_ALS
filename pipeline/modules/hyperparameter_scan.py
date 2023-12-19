from library.packages import *
from modules.model import ALSModel
from modules.model_performance import ModelPerformance
from modules.logger import *

class ALSModelWrapper(BaseEstimator, RegressorMixin):
    """
    This class, ALSModelWrapper, is a wrapper around the Alternating Least Squares (ALS) model, designed to fit into 
    the scikit-learn framework. It inherits from the BaseEstimator and RegressorMixin classes provided by scikit-learn, 
    which provides standard methods for machine learning models. It class used Mean Average Improvement (mai) as a score 
    to evaluate and find the best set of hyperparameter of the model.
    """
    def __init__(
        self,
        nfactors=10,
        regularization=1, 
        alpha=40, 
        iterations=10,
        nthreads=0,
        top_n_products=10,
        sample_size=10,
        evaluation_metric='dot_product',
        a2c_table=pd.DataFrame(),
        imp_table=pd.DataFrame()
    ) -> None:
        """
        Initializes an instance of my class ALSModel.
        Args:
            nfactors (int): initial number of factors for the model,
            regularization (int): initial number for the regulatization factor of the model,
            alpha (int): initial number of the alpha parameter for the model,
            iterations (int): initial number of iterarion when training the model,
            nthreads (int): number of threads to run the algorithm,
            a2c_table (DataFrame): input add to cart table
            imp_table (DataFrame): input impressions table
        """
        self.nfactors = nfactors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.nthreads = nthreads
        self.top_n_products = top_n_products
        self.sample_size = sample_size
        self.evaluation_metric = evaluation_metric
        self.a2c_table = a2c_table
        self.imp_table = imp_table
        self.alsmodel = None
        logger.info('Initializing hyper parameter scan class')
        
    def fit(
        self,
        X: pd.DataFrame, 
        y=None
    ) -> None:
        """
        This method train the ALSModel the instance variables
        Args:
            df (DataFrame): Input dataframe with the information of users, items and affinity_score
        Returns: (store the model in the instance variables self.model of the Class)
            model: model trained
        """
        self.alsmodel = ALSModel(nfactors=self.nfactors,
                                 regularization=self.regularization,
                                 alpha=self.alpha,
                                 iterations=self.iterations,
                                 nthreads=self.nthreads)
        self.alsmodel.train_model(X)

    def score(
        self, 
        X: pd.DataFrame, 
        y=None
    ) -> int:
        """
        This method compute the Mean Average Improvement of the top k products of the test matrix
        Args:
            df (DataFrame): Input dataframe with the information of users, items and affinity_score
        Returns: 
            improvement (int): percentage of improvement
        """
        perform_model = ModelPerformance(model=self.alsmodel,
                                         a2c_table=self.a2c_table,
                                         imp_table=self.imp_table,
                                         top_n_products=self.top_n_products,
                                         sample_size=self.sample_size,
                                         evaluation_metric=self.evaluation_metric)
        perform_model.compute_metric()
        return perform_model.mai