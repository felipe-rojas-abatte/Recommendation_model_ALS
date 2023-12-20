from modules.pipeline import create_matrices, optimize_hyperparameters, train_best_model, evaluate_model_performance, store_latent_vectors, display_final_time
from library.global_variables import TODAY_DATE
from library.packages import *
from modules.logger import *

if __name__ == '__main__':
    # Start Time
    start_time = time.time()
    
    print('************************') 
    print('** Pipeline ALS Model **') 
    print('************************\n') 
    
    # Log Info
    logger.info('Pipeline ALS Model initiated on {}'.format(TODAY_DATE))

    # Data Ingestion
    trainig_matrix, test_matrix, matrix_a2c, matrix_imp = create_matrices()

    # Hyperparameters Optimization
    opt = optimize_hyperparameters(trainig_matrix, matrix_a2c, matrix_imp)

    # Training the Model
    alsmodel_best = train_best_model(opt, trainig_matrix)
    
    # Evaluation and Performance
    eval_alsmodel_best, perform_model_best = evaluate_model_performance(alsmodel_best, test_matrix, matrix_a2c, matrix_imp)

    # Store Latent Vectors
    store_latent_vectors(alsmodel_best)
    
    # Log Info
    logger.info('ALS Model successfully completed')

    # Final Time
    display_final_time(start_time)


