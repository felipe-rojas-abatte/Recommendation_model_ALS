from library.packages import * 
from modules.logger import *

class EvaluationMetrics(): 
    """
    This class is designed to evaluate the performance of a recommendation model using NDCG, MAP and MAR metrics. It takes in a model, 
    a test matrix, a value of k (number of products to test), and a choice of evaluation metric (either dot product or 
    euclidean distance) as inputs. 
    """
    def __init__(
        self,
        model: implicit.cpu.als.AlternatingLeastSquares,
        k: int,
        test_matrix: pd.DataFrame,
        evaluation_metric: str
    ) -> None:
        """
        Initializes an instance of my class EvaluationMetrics.
        Args:
            model (model): input model,
            k (int): initial number for the products you want to test,
            true_rating (DataFrame): input matrix to test model,
            evaluation_metric (str): initial metric user wnat to use to test the model (options are: dot_product/euclidean_distance).
        """
        logger.info('Initializing EvaluationMetrics class')
        self.model = model
        self.k = k
        self.true_rating = test_matrix
        self.evaluation_metric = evaluation_metric
        self.ndcg = None
        self.map = None
        self.mar = None
            
    def __str__(
        self
    ) -> str:
        """
        Returns a string representation with the evaluation of the model.
        Returns:
            str: Provide information of the test matrix: nº of items, nº of users, nº of cells and density, 
                 and also provide information about the evaluation of ther model (ndcg, map and mar).
        """
        total_users = len(self.true_rating['userid'].unique())
        total_items = len(self.true_rating['itemid'].unique())
        total_n_cells = total_users*total_items
        density = round(100*len(self.true_rating)/total_n_cells,2)
        if self.ndcg != None:
            class_info = (
                f"*** Evaluation Model info *** \n"
                f" nº items : {total_items} \n"
                f" nº users : {total_users} \n"
                f" nº cells : {total_n_cells} \n"
                f" density  : {density}% \n"
                f" eval. met: {self.evaluation_metric} \n"
                f" k        : {self.k} \n"
                f" ndcg@k   : {self.ndcg} \n"
                f" map@k    : {self.map} \n"
                f" mar@k    : {self.mar}  "
                )
        else:
            class_info = (
                f" nº items : {total_items} \n"
                f" nº users : {total_users} \n"
                f" nº cells : {total_n_cells} \n"
                f" density  : {density}% "
                )
        return class_info

    def check_vector_existance(
        self
    ) -> pd.DataFrame:
        """
        This method verify if the latent vectors contain all the users and items from the true_rating matrix. If that's the case
        a warning message will be display with the percentage of users and items removed from the true_rating matrix.
        Returns:
            clean_test_matrix/self.true_rating (DataFrame): Clean DataFrame after removing non existent users and items 
        """
        logger.info('Checking vector existence')
        # Create list of unique users and items
        list_users = self.true_rating.userid.unique()    
        list_items = self.true_rating.itemid.unique()
        
        # Create a list of the users and items that do not exist in the letent vectors
        user_not_exist = [user for user in list_users if user not in self.model.users_factors.index]
        item_not_exist = [item for item in list_items if item not in self.model.items_factors.index]
        
        # Create a boolean mask with users and items that do not exist in the letent vectors
        cut_users = (self.true_rating.userid.isin(user_not_exist))
        cut_items = (self.true_rating.itemid.isin(item_not_exist))
   
        if (len(user_not_exist) != 0) or (len(item_not_exist)): 
            print('\n {:.2f}% users were removed !!\n {:.2f}% items were removed !! \n'.format(100*len(user_not_exist)/len(list_users), 
                                                                                               100*len(item_not_exist)/len(list_items)))
            clean_test_matrix = self.true_rating[(~cut_users)&(~cut_items)]
            logger.warning('{:.2f}% users were removed and {:.2f}% items were removed'.format(100*len(user_not_exist)/len(list_users), 
                                                                                              100*len(item_not_exist)/len(list_items)))
            return clean_test_matrix
        else:
            #print('No users or items were removed !!')
            logger.info('No users or items were removed')
            return self.true_rating

    def create_sparse_matrix(
        self,
        df: pd.DataFrame
    ) -> sparse._csr.csr_matrix:
        """
        This method create a sparse matrix from a input dataframe
        Args:
            df (DataFrame): Input dataframe with the information of users, items and affinity_score
        Returns:
            sparse_uim (sparse._csr.csr_matrix): crs sparse matrix array with the information of users, items and affinity_score
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

    def prediction_matrix(
        self,
        matrix: pd.DataFrame
    ) -> np.ndarray:
        """
        This method create a matrix with the predictions of the model from an input DataFrame
        Args:
            matrix (DataFrame): Input test matrix with the information of users, items and affinity_score
        Returns:
            prediction_matrix (np.ndarray): ndarray with the prediction of the model
        """
        logger.info('Creating prediction matrix')
        # Get users_factors and items_factors as numpy arrays
        users_factors = self.model.users_factors.loc[matrix.userid.unique()].values
        items_factors = self.model.items_factors.loc[matrix.itemid.unique()].values
 
        # Matrix multiplication, using broadcasting for dot product
        if self.evaluation_metric == 'dot_product':
            prediction_matrix = np.dot(users_factors, items_factors.T)
        elif self.evaluation_metric == 'euclidean_distance':
            prediction_matrix = distance_matrix(users_factors, items_factors)
            
        logger.info('Prediction matrix with shape {} created using {} metric'.format(prediction_matrix.shape, self.evaluation_metric))
        return prediction_matrix
    
    def prepare_matrix(
        self,
        matrix: pd.DataFrame
    ) -> np.ndarray: 
        """
        This method create two matrices, one with the input DataFrame and the second with the predictions of the model from an input DataFrame
        Args:
            matrix (DataFrame): Input test matrix with the information of users, items and affinity_score
        Returns:
            test_matrix (np.ndarray): ndarray with the test input matrix
            prediction_matrix (np.ndarray): ndarray with the prediction of the model
        """
        logger.info('Preparing test and prediction matrix')
        # Transform matrix into a sparse matrix
        sparse_matrix = self.create_sparse_matrix(matrix)
        # Create a mask with all items selected by users
        #mask = sparse_matrix > 0
        #mask_array = mask.toarray()

        # Create prediction matrix applying mask 
        prediction_matrix = self.prediction_matrix(matrix)
        #prediction_matrix *= mask_array
         
        # Transform sparse matrix into an array    
        test_matrix = sparse_matrix.toarray()
        logger.info('Test and prediction matrix created with shape {} and {} respectively'.format(test_matrix.shape, prediction_matrix.shape))
            
        return test_matrix, prediction_matrix
 
    def get_top_k_items(
        self,
        matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This method select the k top items for each userid sorted by rating
        Args:
            matrix (DataFrame): Input test matrix with the information of users, items and affinity_score
        Returns:
            top_k_items (DataFrame): Dataframe with the itemid sorted by rating for each userid
        """
        logger.info('Getting top {} items'.format(self.k))
        top_k_items = matrix.groupby(['userid']).apply(lambda x: x.nlargest(self.k, 'rating')[['itemid','rating']])
        top_k_items = top_k_items.reset_index(level=1, drop=True)
        return top_k_items

    def sort_itemids(
        self,
        row: pd.Series 
    ) -> pd.Series:
        """
        This method sort the values of each row in descending/ascending order deppending of the input evaluation metric dot_product/euclidean_distance and then assign those values as index in the series
        Args:
            row (Series): Input row
        Returns:
            top_k_items (Series): Sorted row
        """
        if self.evaluation_metric == 'dot_product':
            sorted_items = row.sort_values(ascending=False)
        if self.evaluation_metric == 'euclidean_distance':
            sorted_items = row.sort_values(ascending=True)
            
        return pd.Series(sorted_items.index)    

    def top_k_most_relevat_items(
        self,
        matrix: pd.DataFrame
    ) -> np.ndarray:
        """
        This method select the k top most relevant items for each userid
        Args:
            matrix (DataFrame): Input test matrix with the information of users, items and affinity score
        Returns:
            topk_relevant (ndarray): ndarray with the itemid sorted by rating for each userid
        """
        logger.info('Getting top {} most relevant items'.format(self.k))
        #Create sorted matrix of the top K products from affinity matrix
        df_topk_data = self.get_top_k_items(matrix)
        topk_table = df_topk_data.pivot_table(index='userid',
                                              columns=df_topk_data.groupby('userid').cumcount(),
                                              values='itemid',
                                              aggfunc='first').fillna(0)
        topk_relevant = topk_table.to_numpy()
        return topk_relevant

    def top_k_prediction_items(
        self,
        prediction_matrix: np.ndarray
    ) -> np.ndarray:
        """
        This method select the k top most relevant prediction items for each userid
        Args:
            prediction_matrix (ndarray): Input prediction matrix with the information of users, items and prediction score
        Returns:
            topk_prediction (ndarray): array with the itemid sorted by prediction score for each userid
        """
        logger.info('Getting top {} most relevant predictions items'.format(self.k))
        df_prediction = pd.DataFrame(data=prediction_matrix,
                                     index=self.users_list,
                                     columns=self.items_list)

        # Function that sort items user by user from most to less relevant
        sorted_df = df_prediction.apply(self.sort_itemids, axis=1)
        max_k = min(self.k, prediction_matrix.shape[1])
        topk_prediction = sorted_df.to_numpy()[:,:max_k]
        return topk_prediction
    
    def ndcg_at_k(
        self,
        test_matrix: np.ndarray,
        prediction_matrix: np.ndarray
    ) -> str:
        """
        This method compute the Normalized Discounted Cumulative Gain at top k products. It evaluates the 
        quality of the ranking produced by the model by considering both relevance and the position of items
        in the recommendation list.
        Args:
            test_matrix (ndarray): Input test matrix with the information of users, items and affinity score
            prediction_matrix (ndarray): Input prediction matrix with the information of users, items and predicted score
        Returns:
            ndcg score (str): int value for ndcg
        """
        #logger.info('Computing Normalized Discounted Cumulative Gain')
        self.ndcg = round(ndcg_score(test_matrix, prediction_matrix, k=self.k),4) 
        logger.info('Normalized Discounted Cumulative Gain ndcg@{}: {:.4f}'.format(self.k, self.ndcg))
        
    def map_at_k(
        self,
        topk_relevant: np.ndarray,
        topk_prediction: np.ndarray
    ) -> str:
        """
        This method compute the Mean Average Precision at top k products, where 
           precision is the proportion of relevant items among the k recommended items.
        Args:
            topk_relevant (array): Input array with the top k relevant items per user
            topk_prediction (array): Input array with the top k relevant predictions items per user
        Returns:
            map score: int value for map
        """
        # Initialize the list to keep track of AP scores for each query (user)
        ap_scores = []
        for i in range(topk_relevant.shape[0]):
            # Find the common elements (hits) between the relevant items and the predicted items
            hits = np.isin(topk_prediction[i], topk_relevant[i])
            # Calculate the precision scores for the hits
            precisions = np.cumsum(hits) / (1 + np.arange(len(hits)))
            # Calculate the average precision for the current query
            if np.sum(hits) == 0:
                ap = 0
            else:
                ap = np.sum(precisions * hits) / np.sum(hits)
            # Add the AP score to the list
            ap_scores.append(ap)
        # Calculate MAP by taking the mean of the AP scores
        self.map = round(np.mean(ap_scores),4)
        logger.info('Mean Average Precision map@{}: {}'.format(self.k, self.map))

    def mar_at_k(
        self,
        topk_relevant: np.ndarray,
        topk_prediction: np.ndarray
    ) -> str:
        """
        This method compute the Mean Average Recall at top k products, where recall is the proportion 
        of relevant items that are actually recommended. 
        Args:
            topk_relevant (ndarray): Input array with the top k relevant items per user
            topk_prediction (ndarray): Input array with the top k relevant predictions items per user
        Returns:
            mar score: int value for mar
        """
        #Compute Average Recall metric
        fraction = np.empty(topk_relevant.shape[0])
        # Calculate the common elements between corresponding rows
        common_elements = np.sum(topk_relevant[:, None] == topk_prediction[:, :, None], axis=2)
        len_elements = [sum(1 for element in row if element != 0) for row in topk_relevant]
        # Calculate the fractions of common elements for each row
        fraction = np.sum(common_elements, axis=1) / len_elements
        self.mar = round(fraction.mean(),4)
        logger.info('Mean Average Recall mar@{}: {:.4f}'.format(self.k, self.mar))
         
    def compute_metric(
        self
    ) -> None:
        """
        This method evaluate all metric: ndcg, map, mar
        Returns:
            method ndcg_at_k()
            method map_at_k()
            method mar_at_k()
        """
        print(' Evaluation Model')
        clean_test_matrix = self.check_vector_existance()
        test_matrix, prediction_matrix = self.prepare_matrix(clean_test_matrix)
        topk_relevant = self.top_k_most_relevat_items(clean_test_matrix)
        topk_prediction = self.top_k_prediction_items(prediction_matrix)
        self.ndcg_at_k(test_matrix, prediction_matrix)
        self.map_at_k(topk_relevant, topk_prediction)
        self.mar_at_k(topk_relevant, topk_prediction)