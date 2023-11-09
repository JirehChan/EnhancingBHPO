import time
import logging
import itertools

import my_tool as mtool
import my_cross_validation as mcv
import my_searcher as msearcher
from sklearn.model_selection import cross_val_score

SETTING_PATH = '../settings/'

"""
My Run
"""
def my_run():

    # 1. Read & Set Parameters
    args = mtool.parse_arg()
    mtool.setup_logging(args.dataset_name, args.save_path, args.save_name)

    f = open(SETTING_PATH+'{}/{}.txt'.format(args.set_path, args.set_name), 'r')
    my_args = eval(f.read())
    
    logging.warning('--[START]--')
    logging.warning('- dataset name: {}'.format(args.dataset_name))
    logging.warning('- random seed : {}'.format(args.random_seed))
    logging.warning('- setting path: {}/{}'.format(args.set_path, args.set_name))

    # 2. Load Dataset
    mtool.setup_seed(args.random_seed)
    my_args['data_param']['random_seed'] = args.random_seed

    x_train, y_train, x_val, y_val, x_test, y_test = mtool.load_data(args.dataset_name, **my_args['data_param'])
    
    # 3. Run
    logging.warning('\n--[Runing]--')

    for k, searcher in zip(my_args['searchers'], my_args['searchers'].values()):

        logging.warning('<{}>'.format(k))
        model = mtool.get_model(my_args['model_name'])

        start_time = time.time()
        search_model = msearcher.get_searcher(
            my_args['param_grid'], model, 
            searcher['search_method'], searcher['search_param'],
            searcher['cv_method'], searcher['cv_param'], x_train, y_train)
        end_time = time.time()
        init_time = end_time - start_time
        logging.warning('- Init Time: {}'.format(init_time))

        start_time = time.time()
        search_model.fit(x_train, y_train)
        end_time = time.time()

        logging.warning('- Best Param: {}'.format(search_model.best_params_))
        logging.warning('- Accuracy  : {:5.2f} | {:5.2f} | {:5.2f} '.format(
                search_model.score(x_train, y_train)*100., 
                search_model.score(x_val,y_val)*100., 
                search_model.score(x_test,y_test)*100.))
        logging.warning('- Time      : {}'.format(end_time-start_time))

    
    # 4. Ending
    logging.warning('--[END]--')




if __name__=='__main__':
    import warnings
    warnings.filterwarnings("ignore")

    my_run()