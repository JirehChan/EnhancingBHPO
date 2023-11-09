"""
Load Arguments
"""
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name', default='epsilon', help='the dataset name')
    parser.add_argument(
        '--random_seed', type=int, default=0, help='the random seed')
    parser.add_argument(
        '--is_train', type=int, default=1, help='whether train')

    parser.add_argument(
        '--set_path', default='default', help='the settings')
    parser.add_argument(
        '--set_name', default='default', help='the settings')
    parser.add_argument(
        '--save_path', default='default', help='save the reuslts')
    parser.add_argument(
        '--save_name', default='save.log', help='save the reuslts')

    args, unparsed = parser.parse_known_args()
    return args



"""
Set Random Seed
"""
import random
import numpy as np

def setup_seed(seed):
    np.random.seed(seed) #numpy
    random.seed(seed) #python



"""
Set Logging
"""
import os
import logging

RESULT_PATH = '../result/'

def setup_logging(dataset_name, save_path, save_name, is_visual=False):
    formatter = logging.Formatter("%(message)s")
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level=logging.WARNING)

    if is_visual:
        logs_path = os.path.join(RESULT_PATH+'{}/{}/{}'.format(dataset_name, save_path, save_name), 'visual')
    else:
        logs_path = os.path.join(RESULT_PATH+'{}/{}'.format(dataset_name, save_path), save_name)

    os.makedirs(os.path.dirname(logs_path), exist_ok=True)
    file_handler = logging.FileHandler(logs_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=logging.INFO)
    
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler]) 



"""
Load Dataset
"""
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_data(dataset_name, random_seed=0, test_size=0.3, data_path='../../../../data'):
    
    ## Kaggle
    if dataset_name in ['credit2023','bankfraud2022','fraud','maintenance']:
        df = pd.read_csv("{}/{}/{}/{}.csv".format(data_path, 'kaggle', dataset_name, dataset_name))
        if dataset_name in ['credit2023']:
            df = df.drop(['id'],axis=1)
        elif dataset_name == 'maintenance':
            df = df.drop(['Product ID','UDI','Failure Type'], axis=1)
            
        data_x = df.drop(['Class'],axis=1).values
        data_y = df['Class'].values

        if dataset_name in ['bankfraud2022']:
            data_x = df.drop(['Class'],axis=1)

            s = (data_x.dtypes == 'object')
            object_cols = list(s[s].index)

            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            # Get one-hot-encoded columns
            ohe_cols = pd.DataFrame(ohe.fit_transform(data_x[object_cols]))

            # Set the index of the transformed data to match the original data
            ohe_cols.index = data_x.index

            # Remove the object columns from the training and test data
            num_X = data_x.drop(object_cols, axis=1)

            # Concatenate the numerical data with the transformed categorical data
            data_x = pd.concat([num_X, ohe_cols], axis=1)

            # Newer versions of sklearn require the column names to be strings
            data_x.columns = data_x.columns.astype(str)

            drop_cols = ['zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w']
            data_x.drop(columns=drop_cols, inplace=True)
            data_x = data_x.values


        data_x, x_test, data_y, y_test = train_test_split(
            data_x, data_y, test_size=test_size, random_state=random_seed)
    ## UCI
    elif dataset_name in ['tunadromd', 'naticusdroid','student']:
        df = pd.read_csv("{}/{}/{}/{}.csv".format(data_path, 'uci', dataset_name, dataset_name))
        data_x = df.drop(['Label'],axis=1).values
        data_y = df['Label'].values
        data_x, x_test, data_y, y_test = train_test_split(
            data_x, data_y, test_size=test_size, random_state=random_seed)
    ## LibSVM
    elif dataset_name in ['australian', 'colon-cancer', 'ionosphere']:
        data_x, data_y = datasets.load_svmlight_file(
                    ("{}/{}/{}/{}".format(data_path, 'libsvm', dataset_name, dataset_name)))
        data_x, x_test, data_y, y_test = train_test_split(
            data_x, data_y, test_size=test_size, random_state=random_seed)

    else:
        data_x, data_y, x_test, y_test = datasets.load_svmlight_files(
            ("{}/{}/{}/train".format(data_path, 'libsvm', dataset_name), 
             "{}/{}/{}/test".format(data_path, 'libsvm', dataset_name)))
    
    if dataset_name in ['a1a', 'a9a', 'gisette', 'splice', 'epsilon', 'australian', 'colon-cancer', 'ionosphere', 'credit2023', 'tunadromd', 'naticusdroid', 'student', 'bankfraud2022', 'fraud', 'maintenance']:
        data_y = data_y.clip(0)+1
        y_test = y_test.clip(0)+1

    if dataset_name not in ['credit2023', 'tunadromd', 'naticusdroid', 'student', 'bankfraud2022', 'fraud', 'maintenance']:
        data_x = data_x.toarray()
        x_test  = x_test.toarray()
        
    indexs = np.argsort(np.array(data_y).astype(int))
    data_x = data_x[indexs]
    data_y = data_y[indexs]
    n_class = int(data_y[-1])
    

    x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=test_size, random_state=random_seed)
    
    return x_train, y_train, x_val, y_val, x_test, y_test





"""
Get Model
"""
def get_model(model_name):
    if model_name=='RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
    elif model_name=='MLPClassifier':
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier()
    elif model_name=='SVC':
        from sklearn.svm import SVC
        return SVC()
    elif model_name=='GradientBoostingClassifier':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier()
    else:
        logging.warning('Do not support such model ({}), please check your input.'.format(model_name))