from library.packages import * 
from library.global_variables import QUERY_FOLDER_PATH
from modules.logger import *

class DataIngestion():
    """
    The DataIngestion class is used to handle data ingestion and processing from an SQL database. 
    It provides several methods to load data, preprocess it, and then return relevant information.
    """
    def __init__(
        self
    ) -> None:
        """
        Initializes an instance of my class DataIngestion.
        Args:
            init_date (str): Initial date of the period to obtain the data
            end_date (str): Final date of the period to obtain the data
            query_file (str): sql query for the instance
            matrix (DataFrame): DataFrame that hold the information from affinity score / add to cart / impressions, 
            depending of the query_file used.
        """
        logger.info('Initializing Data Ingestion class')
        self.init_date = None
        self.end_date = None
        self.query_file = None
        self.a2c_query_file = None
        self.imp_query_file = None
        self.matrix = None
        
    def __str__(
        self
    ) -> str:
        """
        Returns a string representation of the instance.
        Returns:
            str: Provide information of the affinity matrix with the nº of items, 
                 nº of users, nº of cells and density.
        """
        date_format = "%Y%m%d"
        init_date = datetime.strptime(self.init_date, date_format)
        end_date = datetime.strptime(self.end_date, date_format)
        delta = end_date - init_date
        total_users = len(self.matrix['userid'].unique())
        total_items = len(self.matrix['itemid'].unique())
        total_n_cells = total_users*total_items
        density = round(100*len(self.matrix)/total_n_cells,2)
        class_info = (
            f"*** Matrix info *** \n"
            f" matrix    : {self.query_file} \n"
            f" init date : {init_date} \n"
            f" end date  : {end_date} \n"
            f" nº days   : {delta.days} \n"
            f" nº items  : {total_items} \n"
            f" nº users  : {total_users} \n"
            f" nº cells  : {total_n_cells} \n"
            f" density   : {density}% \n"
            )
        return class_info
    
    def get_query_path(
        self
    ) -> str:
        """
        Method that generate the path to the query inside the query folder
        Returns:
            query_path (str): Path to the query
        """
        try:
            logger.info('Getting query path')
            query_path = os.path.join(QUERY_FOLDER_PATH, self.query_file)
            logger.info(f'Query path: {query_path}')
            return query_path
        except Exception as e:
            logger.error(f'Failed to get query path: {e}')
            raise
            
        #query_path = os.path.join(QUERY_FOLDER_PATH, self.query_file)
        #return query_path
        
    def get_query(
        self, 
        query_path 
    ) -> str:
        """
        This method replace the init_date and end_date for the sql query
        Args: 
            query_path (str): Path of the sql query
        Returns: 
            query (str): str with the full sql query
        """
        try:
            logger.info('Reading query from file')
            with open(query_path) as f:
                query = f.read()
                if self.init_date:
                    query = query.replace(':init_date', self.init_date)
                if self.end_date:
                    query = query.replace(':end_date', self.end_date)
            logger.info('Query successfully read and modified')
            return query
        except FileNotFoundError:
            logger.error(f'File {query_path} not found')
            raise
        except Exception as e:
            logger.error(f'Failed to read query: {e}')
            raise
    
    def ingest_data(
        self
    ) -> pd.DataFrame:
        """
        This method generate a dataframe with the information provided by a query.
        Returns:
            df (DataFrame): DataFrame with the information of users, items and affinity_score
        """
        try:
            logger.info(
                f'Ingesting data from: {self.query_file} - '
                f'Init date: {self.init_date} - '
                f'End date : {self.end_date}'
            )
            query = self.get_query(self.get_query_path())
            logger.info('Running query on client')
            matrix = client.query(query).result()
            df = matrix.to_dataframe()
            logger.info('Data ingested successfully.')
            return df
        except Exception as e:
            logger.error(f'Failed to ingest data: {e}')
            raise
    
    def get_user_item_matrix(
        self, 
        query_file: str,
        init_date: str,
        end_date: str
    ) -> None:
        """
        This method create dataframe from the input query and preprocess the dataframe removing duplicates and nan values
        Args: (store the query in the instance variable self.query of the Class)
            query (str): Input sql query
        Returns: (store the DataFrame in the instance variable self.matrix of the Class)
            self.matrix (DataFrame): DataFrame with processed the information of users, items and affinity_score
        """
        self.query_file = query_file
        self.init_date = init_date
        self.end_date = end_date
        df = self.ingest_data()
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df = df.rename(columns={'affinity_score':'rating'})
        self.matrix = df
             
    def get_add2cart(
        self,
        query_file: str,
        init_date: str
    ) -> pd.DataFrame:
        """
        This method create dataframe with the add2cart information of users from the input query
        Args: (store the query in the instance variable self.query of the Class)
            query (str): Input sql query
        Returns: (store the DataFrame in the instance variable self.matrix of the Class)
            DataFrame: DataFrame with processed the information of add2cart
        """
        self.query_file = query_file
        self.init_date = init_date
        a2c_data = self.ingest_data()
        # Rename columns of the add2cart data
        a2c_data = a2c_data.rename(columns={'item_list_index':'a2c_position','item_id':'a2c_itemid'})
        a2c_data = a2c_data[['fecha','search_query','userid','correlation_id','a2c_itemid','a2c_position']]
        # Remove indefined users
        a2c_data = a2c_data[~a2c_data['userid'].isin(['undefined'])]
        # Remove indefined items
        a2c_data = a2c_data[~a2c_data['a2c_itemid'].isin(['(not set)'])]
        # Remove duplicates values
        a2c_data.drop_duplicates(inplace=True)
        self.matrix = a2c_data

    def get_impression(
        self, 
        query_file: str,
        init_date: str
    ) -> pd.DataFrame:
        """
        This method create dataframe with the list impression information of users from the input query
        Args: (store the query in the instance variable self.query of the Class)
            query (str): Input sql query
        Returns: (store the DataFrame in the instance variable self.matrix of the Class)
            DataFrame: DataFrame with processed the information of impression
        """
        self.query_file = query_file
        self.init_date = init_date
        imp_data = self.ingest_data()
        # Rename columns of the impression data
        imp_data = imp_data.rename(columns={'SCL_CREATED_DATE':'fecha',
                                            'customer_id':'userid',
                                            'response_sku':'imp_itemid',
                                            'response_position':'imp_position'})
        imp_data = imp_data[['fecha','userid','correlation_id','imp_itemid','imp_position']]
        # substract 1 position to match with add2cart table
        imp_data['imp_position'] = imp_data['imp_position']-1.0
        # Remove indefined users
        imp_data = imp_data[~imp_data['userid'].isin(['undefined'])]
        self.matrix = imp_data    