from library.packages import *
from modules.logger import *

class ModelPerformance():
    """
    The ModelPerformance class is used to evaluate the performance of an ALS (Alternating Least Squares) model. ALS is a 
    recommendation algorithm used for collaborative filtering. This class computes various metrics to assess the model's 
    performance, such as the Mean Average Improvement (mai) and Mean Average Number of Improvement Positions (manip).
    """ 
    def __init__(
        self, 
        model: implicit.cpu.als.AlternatingLeastSquares, 
        a2c_table: pd.DataFrame, 
        imp_table: pd.DataFrame, 
        top_n_products: int, 
        sample_size: int, 
        evaluation_metric: str
    ) -> None:
        """
        Initializes an instance of my class ModelPerformance.
        Args:
            model (model): initial model,
            a2c_table (dataframe): initial input dataframe with information of the add 
                                   to carts of several clients of a specific day,
            imp_table (dataframe): initial input dataframe with information of the impressions 
                                   of several clients of a specific day,
            top_n_products (int): initial number of products to check,
            sample_size (int): initial number of the size of the sample that the class is going 
                               to compute for each product (between 0.0 and 1.0).
            evaluation_metric (str): initial metric user wnat to use to test the model (options are: dot_product/euclidean_distance).
        """
        logger.info('Initializing ModelPerformance class with top_n_products={}, sample_size={}% and evaluation_metric={}'.format(top_n_products, 100*sample_size, evaluation_metric))
        self.model = model
        self.a2c_table = a2c_table
        self.imp_table = imp_table
        self.top_n_products = top_n_products
        self.sample_size = sample_size
        self.evaluation_metric = evaluation_metric
        self.merged_sample_table = None
        self.result = None
        self.mai = None
        self.manip = None
        self.result_by_product = None
        self.q10_val = None
        self.q25_val = None
        self.q50_val = None
        self.q75_val = None
        self.q90_val = None
        
    def __str__(
        self
    ) -> str:
        """
        Returns a string representation of the instance.
        Returns:
            str: Provide information of the instance: top k products, sample size, mean average improvement and mean 
                 average number of improvement positions.
        """
        class_info = (
            f"*** Performance Model info *** \n"
            f" top {self.top_n_products} products \n"
            f" sample size : {100*self.sample_size}% \n"
            f" eval. met.  : {self.evaluation_metric} \n"  
            f" mai@k       : {self.mai} \n"
            f" manip@k     : {self.manip} \n"
            f" q10         : {self.q10_val} \n"
            f" q25         : {self.q25_val} \n"
            f" q50         : {self.q50_val} \n"
            f" q75         : {self.q75_val} \n"
            f" q90         : {self.q90_val} "
            )
        return class_info   
    
    def sample_a2c(
        self
    ) -> pd.DataFrame:
        """
        This method compute a sample dataframe per each search_query gropup and create a new Dataframe with the sample
        Returns:
            a2c_grouped_sample (DataFrame): Sample Dataframe from the original
        """
        a2c_grouped_sample = self.a2c_table.groupby(['search_query']).apply(lambda x: x.sample(n=int(len(x)*self.sample_size), 
                                                                                               replace=True,
                                                                                               random_state=1))
        a2c_grouped_sample = a2c_grouped_sample.reset_index(drop=True)
        return a2c_grouped_sample
        
    def merge_a2c_imp(
        self
    ) -> None:
        """
        This method compute a sample dataframe with the top k products added to a cart and then merge that 
        dataframe with the impressions
        Returns: (store the DataFrame in the instance variables self.merged_sample_table of the Class)
            self.merged_sample_table (DataFrame): Merged dataframe between add2cart and impressions
        """
        a2c_grouped = self.sample_a2c()
        list_of_products = a2c_grouped.groupby(['search_query'])[['userid']].count().rename(columns={'userid':'count'}).sort_values(by=['count'], ascending=False).reset_index()
        top_k_a2c = a2c_grouped[a2c_grouped['search_query'].isin(list_of_products['search_query'][:self.top_n_products])]
        df = pd.merge(self.imp_table, top_k_a2c, on=('fecha','userid','correlation_id'), how='inner')
        df = df.sort_values(by=['fecha','userid','correlation_id','a2c_itemid','imp_position'], ascending=[True,True,True,True,True])
        df = df.reset_index(drop=True)
        self.merged_sample_table = df

    def create_new_matrix_vector(
        self,
        lst: List[str],
        latent_vector: pd.DataFrame 
    ) -> np.ndarray:
        """
        This method create a matrix with the values of the latent vectors according to the input list, keeping that order
        Args:
            lst (List[str]): list of strings with information of either users or items
        Returns:
            latent_vector_array (np.ndarray): ndarray with information of the latent vectors based on the input list
        """
        empty_latent_vector = np.matrix(-1.0*np.array(np.ones(latent_vector.shape[1])))
        latent_vector_array = [latent_vector.loc[ids].values if ids in latent_vector.index else empty_latent_vector for ids in lst]
        latent_vector_array = np.array(latent_vector_array)
        latent_vector_array = latent_vector_array.reshape(len(lst), latent_vector.shape[1])
        return latent_vector_array
    
    def compute_prediction(
        self
    ) -> None:
        """
        This method compute the dot product between the user and the items matrix and store the prediction as a new colum of
        the self.merged_sample_table DataFrame
        """
        list_users = self.merged_sample_table.userid.tolist()
        list_items = self.merged_sample_table.imp_itemid.tolist()
        
        users_factors = self.create_new_matrix_vector(list_users, self.model.users_factors)
        items_factors = self.create_new_matrix_vector(list_items, self.model.items_factors)        
        
        score = []
        if self.evaluation_metric == 'dot_product':
            for i in range(users_factors.shape[0]):
                score.append(np.dot(users_factors[i], items_factors[i].T))
        if self.evaluation_metric == 'euclidean_distance':
            for i in range(users_factors.shape[0]):
                score.append(np.linalg.norm(users_factors[i]-items_factors[i]))
                
        self.merged_sample_table['prediction'] = np.array(score)
    
    def sort_prediction(
        self
    ) -> None:
        """
        This method sort only the 'imp_itemid','prediction' columns of the self.merged_sample_table DataFrame 
        """
        if self.evaluation_metric == 'dot_product':
            df_sorted = self.merged_sample_table.groupby(['fecha',
                                                          'userid',
                                                          'correlation_id',
                                                          'search_query',
                                                          'a2c_itemid'])[['imp_itemid',
                                                                          'prediction']].apply(lambda x: x.sort_values(by=x.columns[-1], ascending=False))
        if self.evaluation_metric == 'euclidean_distance':
            df_sorted = self.merged_sample_table.groupby(['fecha',
                                                          'userid',
                                                          'correlation_id',
                                                          'search_query',
                                                          'a2c_itemid'])[['imp_itemid',
                                                                          'prediction']].apply(lambda x: x.sort_values(by=x.columns[-1], ascending=True))
            
        df_sorted = df_sorted.reset_index(drop=True)
        df_sorted = df_sorted.rename(columns={'imp_itemid':'predict_itemid'})
        self.merged_sample_table = pd.concat([self.merged_sample_table, df_sorted], axis=1)
        
    def imp_at_k(
        self
    ) -> None:
        """
        This method compute the Different Position between the current and predicted position of the sample
        Returns:
            method mean_imp_t_k()
            method mean_imp_at_k_by_product()        
        """
        cut_impresion = (self.merged_sample_table['imp_itemid'] == self.merged_sample_table['a2c_itemid'])
        cut_prediction = (self.merged_sample_table['predict_itemid'] == self.merged_sample_table['a2c_itemid'])
        
        impresion = self.merged_sample_table[cut_impresion][['fecha','userid','correlation_id','search_query','a2c_itemid','imp_itemid','imp_position']]
        predicted = self.merged_sample_table[cut_prediction][['fecha','userid','correlation_id','search_query','a2c_itemid','predict_itemid','imp_position']]
        result = pd.merge(impresion, predicted, on=('fecha','userid','correlation_id','search_query','a2c_itemid'), how='inner').rename(columns={'imp_position_x':'current_pos','imp_position_y':'predict_pos'})
        result['diff_pos'] = result['current_pos']-result['predict_pos']
        self.result = result
        self.mai_at_k()
        self.mai_at_k_by_product()
        
    def mai_at_k( 
        self
    ) -> str:
        """
        This method compute the Mean Average Improvement (mai) and Mean Average Number of Improvement Positions (manip) of the top k products 
        selected previously are because of the prediction of the model
        Returns:
            result (str)       
        """
        df_statistics = self.result.aggregate({'diff_pos':['count', 
                                                           'mean', 
                                                           self.q10, 
                                                           self.q25, 
                                                           self.q50, 
                                                           self.q75, 
                                                           self.q90],
                                               'current_pos':['mean'],
                                               'predict_pos':['mean']})
        df_statistics['mai'] = round(df_statistics['diff_pos']/df_statistics['current_pos'],3)
        self.mai = df_statistics['mai']['mean']
        self.manip = round(df_statistics['diff_pos']['mean'],2)
        self.q10_val = df_statistics['diff_pos']['q10']
        self.q25_val = df_statistics['diff_pos']['q25']
        self.q50_val = df_statistics['diff_pos']['q50']
        self.q75_val = df_statistics['diff_pos']['q75']
        self.q90_val = df_statistics['diff_pos']['q90']
        logger.info('Completed calculation of mai@{}: {}, manip@{}: {}'.format(self.top_n_products, self.mai, self.top_n_products, self.manip))

    def mai_at_k_by_product(
        self
    ) -> None:
        """
        This method compute the Mean Average Improvement per gruop of search query of the top k products 
        Returns:
            result_by_product (DataFrame): DataFrame with the result per product       
        """
        logger.info('Computing Mean Average Improvement by product')
        df_statistics_by_product = self.result.groupby(['search_query']).aggregate({'diff_pos':['count', 
                                                                                                'mean', 
                                                                                                self.q10, 
                                                                                                self.q25, 
                                                                                                self.q50, 
                                                                                                self.q75, 
                                                                                                self.q90],
                                                                                                'current_pos':['mean'],
                                                                                                'predict_pos':['mean']})
        df_statistics_by_product['mai'] = round(df_statistics_by_product['diff_pos']['mean']/df_statistics_by_product['current_pos']['mean'],2)
        df_statistics_by_product = df_statistics_by_product.sort_values(by=['mai'], ascending=False)
        self.result_by_product = df_statistics_by_product
        logger.info('Completed calculation of Mean Average Improvement by product')
    
    def q10(self, x):
        return x.quantile(0.1)
    def q25(self, x):
        return x.quantile(0.25)
    def q50(self, x):
        return x.quantile(0.5)
    def q75(self, x):
        return x.quantile(0.75)
    def q90(self, x):
        return x.quantile(0.9)

    def compute_metric(self):
        self.merge_a2c_imp()
        self.compute_prediction()
        self.sort_prediction()
        self.imp_at_k()