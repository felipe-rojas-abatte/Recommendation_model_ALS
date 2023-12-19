from library.packages import *
from library.global_variables import USERS_FACTORS_NAME, ITEMS_FACTORS_NAME, LATENT_VECTORS_STORING_PATH
from modules.logger import *

class ALSModel():
    
    def __init__(
        self, 
        nfactors: int, 
        regularization: int, 
        alpha: int, 
        iterations: int,
        nthreads: int  
    ) -> None:
        """
        Initializes an instance of my class ALSModel.
        Args:
            nfactors (int): initial number of factors for the model,
            regularization (int): initial number for the regulatization factor of the model,
            alpha (int): initial number of the alpha parameter for the model,
            iterations (int): initial number of iterarion when training the model,
            nthreads (int): number of threads to run the algorithm.
        """
        logger.info('Initializing ALSModel class')
        self.nfactors = nfactors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.nthreads = nthreads
        self.model = None
        self.users_list = None
        self.items_list = None
        self.users_factors = None
        self.items_factors = None
        logger.info(
            f'Parameter inputs: nfactors={self.nfactors}, '
            f'regularization={self.regularization}, '
            f'alpha={self.alpha}, '
            f'iterations={self.iterations}, '
            f'nthreads={self.nthreads}'
        )
        
    def __str__(
        self
    ) -> str:
        """
        Returns a string representation of the model.
        Returns:
            str: Provide information of the initial parameters of the model and once the model is trained it provide
                 information of the latent vector shape.
        """
        if self.model != None:
            model_info = (
                f"*** Alternating Least Square Model *** \n"
                f"  - nº factors    : {self.nfactors} \n"
                f"  - regularization: {self.regularization} \n"
                f"  - alpha         : {self.alpha} \n"
                f"  - nº iterations : {self.iterations} \n"
                f"  - nº threads    : {self.nthreads} \n"
                f"  - latent vectors: \n"
                f"        users factors = users: {self.users_factors.shape[0]}, features: {self.users_factors.shape[1]} \n"
                f"        items factors = items: {self.items_factors.shape[0]}, features: {self.items_factors.shape[1]} "
                )
        else:
            model_info = (
                f"*** Alternating Least Square Model *** \n"
                f"  - nº factors    : {self.nfactors} \n"
                f"  - regularization: {self.regularization} \n"
                f"  - alpha         : {self.alpha} \n"
                f"  - nº iterations : {self.iterations} "
                f"  - nº threads    : {self.nthreads} \n"
                )
        return model_info
            
    def create_sparse_matrix(
        self,
        df: pd.DataFrame
    ) -> sparse._csr.csr_matrix:
        """
        This method create a sparse matrix from a input dataframe
        Args:
            df (DataFrame): Input dataframe with the information of users, items and affinity_score
        Returns:
            sparse_uim (sparse._csr.csr_matrix): csr sparse matrix array with the information of users, items and affinity_score
        """
        logger.info('Creating Sparse user-item matrix')
        # Create list of unique users and items
        self.users_list = list(df.userid.unique())
        self.items_list = list(df.itemid.unique())
        
        # Create numerical index for users and items
        users = pd.factorize(df.userid)[0]
        items = pd.factorize(df.itemid)[0]
        
        #sparse user-item matrix
        sparse_uim = sparse.csr_matrix((df['rating'], (users, items)),
                                        shape=(len(self.users_list), len(self.items_list)))
        sparse_uim = sparse_uim.astype('double')
        logger.info('Sparse user-item matrix created with shape: {}'.format(sparse_uim.shape))
        return sparse_uim
            
    def get_latent_vectors(
        self
    ) -> None:
        """
        This method create latent vectors of dimension nº users/nº items and nfactors
        Returns: (store the latent vectors in the instance variables self.users_factors/self.items_factors of the Class)
            DataFrame: Latent vectors for users and items
        """
        if self.model is None:
            logger.error('Cannot get latent vectors because model has not been trained yet.')
            
        self.users_factors = pd.DataFrame(self.model.user_factors, columns = ['c{}'.format(i) for i in range(1,self.nfactors+1,1)])
        self.users_factors.index = [self.users_list]
        logger.info('Users latent vectors created with shape: {}'.format(self.users_factors.shape))
           
        self.items_factors = pd.DataFrame(self.model.item_factors, columns = ['c{}'.format(i) for i in range(1,self.nfactors+1,1)])
        self.items_factors.index = [self.items_list]
        logger.info('Items latent vectors created with shape: {}'.format(self.items_factors.shape))

    def train_model(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        This method create model and latent vectors from an input dataframe
        Args:
            df (DataFrame): Input dataframe with the information of users, items and affinity_score
        Returns: (store the model and latent vectors in the instance variables 
                 self.model/self.users_factors/self.items_factors of the Class)
            model: model trained
            method get_latent_vectors()
        """
        # Transform a dataframe into a sparse matrix
        sparse_uim = self.create_sparse_matrix(df)
        
        # Initiate the model class with the input parameters
        model = implicit.als.AlternatingLeastSquares(factors=self.nfactors,
                                                     regularization=self.regularization,
                                                     alpha=self.alpha,
                                                     iterations=self.iterations,
                                                     calculate_training_loss=True,
                                                     num_threads=self.nthreads,
                                                     use_native=True,
                                                     use_cg=True)
        logger.info('Training model')
        # Train model with sparse matrix as input
        print(' Trainig model')
        model.fit(sparse_uim)
        self.model = model
        logger.info('Model training complete')
        # Get latent vectors from the model
        self.get_latent_vectors()
        
    def store_latent_vectors(
        self
    ) -> pd.DataFrame:
        """
        This method store the latent vectors of users and items into a dataframe.
        """
        if (self.users_factors is None) or (self.items_factors is None):
            logger.error('Cannot store latent vectors because they have not been created yet.')
            
        self.users_factors.to_csv(USERS_FACTORS_NAME)
        self.items_factors.to_csv(ITEMS_FACTORS_NAME)
        
        vectors = [USERS_FACTORS_NAME, ITEMS_FACTORS_NAME]
        
        for file in vectors:
            new_path = LATENT_VECTORS_STORING_PATH +'/'+ file
            shutil.move(file, new_path)
        
        logger.info('Latent vectors stored in {}'.format(LATENT_VECTORS_STORING_PATH))
    