from library.packages import * 

current_dir = os.getcwd()

QUERY_FOLDER_PATH = current_dir+'/sql_queries/'
LATENT_VECTORS_STORING_PATH = current_dir+'/latent_vectors/'
TODAY_DATE = date.today()
USERS_FACTORS_NAME = 'latent_vector_users_'+str(TODAY_DATE)+'.csv'
ITEMS_FACTORS_NAME = 'latent_vector_items_'+str(TODAY_DATE)+'.csv'