from library.packages import *
from library.global_variables import TODAY_DATE
from modules.data_ingestion import DataIngestion
from modules.hyperparameter_scan import ALSModelWrapper
from modules.model import ALSModel
from modules.evaluation_metric import EvaluationMetrics
from modules.model_performance import ModelPerformance
from modules.logger import *
import parameters

def create_matrices():
    print('Creating training matrix \n')
    trainig_matrix = create_matrix('train_matrix', 'init_train_date', 'end_train_date')
    print(trainig_matrix)
    print('Creating test matrix \n')
    test_matrix = create_matrix('test_matrix', 'test_date', 'test_date')
    print(test_matrix)
    print('Creating Add2cart matrix \n')
    matrix_a2c = create_matrix('a2c_matrix', 'test_date')
    print('Creating Impressions matrix \n')
    matrix_imp = create_matrix('imp_matrix', 'test_date')
    return trainig_matrix, test_matrix, matrix_a2c, matrix_imp

def create_matrix(query_type, init_date, end_date=None):
    matrix = DataIngestion()
    if (query_type == 'train_matrix') or (query_type=='test_matrix'):
        matrix.get_user_item_matrix(query_file=parameters.queries_parameters[query_type][0],
                                    init_date=parameters.date_parameters[init_date][0], 
                                    end_date=parameters.date_parameters[end_date][0])
    elif (query_type == 'a2c_matrix'):
        matrix.get_add2cart(query_file=parameters.queries_parameters[query_type][0],
                            init_date=parameters.date_parameters[init_date][0])
    elif (query_type == 'imp_matrix'):
        matrix.get_impression(query_file=parameters.queries_parameters[query_type][0],
                              init_date=parameters.date_parameters[init_date][0])
    return matrix


def optimize_hyperparameters(trainig_matrix, matrix_a2c, matrix_imp):
    print('Finding best hyperparameters \n')
    opt = BayesSearchCV(
        ALSModelWrapper(iterations=parameters.scan_parameters_model['iterations'][0],
                        nthreads=parameters.bayessearchcv_parameters['n_jobs'][0],
                        top_n_products=parameters.evaluation_parameters['top_n_products'][0],
                        sample_size=parameters.evaluation_parameters['sample_size'][0],
                        evaluation_metric=parameters.evaluation_parameters['evaluation_metric'][0],
                        a2c_table=matrix_a2c.matrix,
                        imp_table=matrix_imp.matrix),
        {
            'nfactors': Integer(
                                parameters.scan_parameters_model['nfactors'][0], 
                                parameters.scan_parameters_model['nfactors'][1]
                            ),
            'regularization': Integer(
                                      parameters.scan_parameters_model['regularization'][0], 
                                      parameters.scan_parameters_model['regularization'][1]
                            ),
            'alpha': Integer(
                             parameters.scan_parameters_model['alpha'][0], 
                             parameters.scan_parameters_model['alpha'][1]
                            )
        },
        n_iter=parameters.bayessearchcv_parameters['n_iter'][0],
        cv=parameters.bayessearchcv_parameters['cv'][0],
        n_jobs=parameters.bayessearchcv_parameters['n_jobs'][0],
        verbose=parameters.bayessearchcv_parameters['verbose'][0]
    )
    
    logger.info(
            f"Range of hyperparameters scan: nfactors=[{parameters.scan_parameters_model['nfactors'][0]}-{parameters.scan_parameters_model['nfactors'][1]}], "
            f"regularization=[{parameters.scan_parameters_model['regularization'][0]}-{parameters.scan_parameters_model['regularization'][1]}], "
            f"alpha=[{parameters.scan_parameters_model['alpha'][0]}-{parameters.scan_parameters_model['alpha'][1]}], "
            f"iterations={parameters.scan_parameters_model['iterations'][0]}, "
            f"nº scan points={parameters.bayessearchcv_parameters['n_iter'][0]}, "
            f"nº threads={parameters.bayessearchcv_parameters['n_jobs'][0]} "
        )
    
    opt.fit(trainig_matrix.matrix)
    print("Best parameters found: ", opt.best_params_)
    print("Best mai found: ", opt.best_score_)
    results_opt = pd.DataFrame(opt.cv_results_)
    results_opt.to_csv('ALS_hyperparameter_scan_'+str(TODAY_DATE)+'.csv')
    return opt


def train_best_model(opt, trainig_matrix):
    print('Training best Model \n')
    alsmodel_best = ALSModel(
                             nfactors=opt.best_params_['nfactors'], 
                             regularization=opt.best_params_['regularization'], 
                             alpha=opt.best_params_['alpha'], 
                             iterations=parameters.scan_parameters_model['iterations'][0],
                             nthreads=parameters.bayessearchcv_parameters['n_jobs'][0]
                            )
    alsmodel_best.train_model(trainig_matrix.matrix)
    print(alsmodel_best)
    return alsmodel_best


def evaluate_model_performance(alsmodel_best, test_matrix, matrix_a2c, matrix_imp):
    print('Evaluating Model and Performance \n')
    eval_alsmodel_best = EvaluationMetrics(
                                           model=alsmodel_best, 
                                           k=parameters.evaluation_parameters['cutoff_k'][0], 
                                           test_matrix=test_matrix.matrix, 
                                           evaluation_metric=parameters.evaluation_parameters['evaluation_metric'][0]
                                          )
    eval_alsmodel_best.compute_metric()
    print(eval_alsmodel_best)
    
    perform_model_best = ModelPerformance(
                                          model=alsmodel_best, 
                                          a2c_table=matrix_a2c.matrix, 
                                          imp_table=matrix_imp.matrix, 
                                          top_n_products=parameters.evaluation_parameters['top_n_products'][0], 
                                          sample_size=parameters.evaluation_parameters['sample_size'][0],
                                          evaluation_metric=parameters.evaluation_parameters['evaluation_metric'][0]
                                         )
    perform_model_best.compute_metric()
    print(perform_model_best)
    
    return eval_alsmodel_best, perform_model_best


def store_latent_vectors(alsmodel_best):
    print('Storing Latent Vectors \n')
    alsmodel_best.store_latent_vectors()


def display_final_time(start_time):
    final_time = time.time() - start_time
    if final_time <= 60.0:
        print('Final time = {:.2f} seconds'.format(final_time))
    elif (final_time > 60.0) & (final_time <= 3600.0):
        print('Final time = {:.2f} minutes'.format(final_time/60.0))
    elif final_time > 3600.0:
        print('Final time = {:.2f} hours'.format(final_time/3600.0))