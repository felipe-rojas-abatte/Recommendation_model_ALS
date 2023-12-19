#range of date
date_parameters = dict(
    {
     'init_train_date':['20221201'],
     'end_train_date':['20231201'],
     'test_date':['20231205']   
    }
)

queries_parameters = dict(
    {
     #query add2cart   
     'a2c_matrix':['add2cart_test.sql'],
     #query impressions   
     'imp_matrix':['impressions_test.sql'],
     #query train matrix      
     'train_matrix':['affinity_matrix.sql'],
     #query test matrix      
     'test_matrix':['affinity_matrix.sql']
    }
)
     
#model parameters for scan
scan_parameters_model = dict(
    {
     'nfactors':[10, 50],
     'regularization':[1, 30],
     'alpha':[40, 80],
     'iterations':[10]
    }
) 
#evaluation parameters   
evaluation_parameters = dict(
    {
     'cutoff_k':[20],   
     'evaluation_metric':['dot_product'],
     'top_n_products':[100],
     'sample_size':[0.2]
    }
)
#hyperparameters of bayes search cv
bayessearchcv_parameters = dict(
    {
     'n_iter':[20],
     'cv':[3],
     'n_jobs':[8],
     'verbose':[2]  
    }
)