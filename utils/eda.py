import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

####################  Univariate Variable  ###################
def check_target(df, target_col = 'TARGET', bad_value = 1, good_value = 0):
    '''
    Check target distribution
    '''
    print('Number of missing value : ', df[target_col].isnull().sum())
    print('Number of value not good or bad : ', (~df[target_col].isin([bad_value, good_value])).sum())
    print('Number of bads is : ', (df[target_col]==bad_value).sum())
    print('Number of goods is : ', (df[target_col]==good_value).sum())
    print('bad rate is : {:.2%}'.format(df[target_col].mean()))

def get_num_cat_list(df, target_col = 'TARGET', id_col = 'SK_ID_CURR', date_col_keyword = 'DATE'):
    '''
    Get a list of numeric and a list of categorical features
    '''
    print('Start with {} total columns'.format(df.shape[1]))
    target_col = ['TARGET']
    print('Removing taret column: ', target_col)
    id_col = ['SK_ID_CURR']
    print('Removing ID columns: ', id_col)
    date_cols = [x for x in df.columns if 'DATE' in str.upper(x)] + df.select_dtypes(include=[np.datetime64]).columns.tolist()
    print('Removing date type columns: ', date_cols)
    uniform_value_col = df.nunique()[df.nunique()==1].index.tolist()
    print('Removing uniform value columns: ', uniform_value_col)
    dup_col = df.columns[df.columns.duplicated()].tolist()
    print('Removing duplicated columns: ', dup_col)

    cols_to_remove = target_col + date_cols + id_col + uniform_value_col + dup_col
    eligible_cols = [x for x in df.columns if x not in cols_to_remove]
    print('Ended with {} eligible columns'.format(len(eligible_cols)))

    num_cols = df[eligible_cols].select_dtypes(include=np.number).columns.tolist()
    print('{} numeric columns'.format(len(num_cols)))

    cat_cols = df[eligible_cols].select_dtypes(exclude=np.number).columns.tolist()
    print('{} categorical or mixed type columns'.format(len(cat_cols)))
    
    return num_cols, cat_cols

def describe_num_col(df, num_cols):
    '''
    Get descriptive stats of numeric columns - quantiles, missing rate, number of unique values
    '''
    num_col_summary = df[num_cols].describe().transpose()
    num_col_summary['N_missing_perc'] = (df[num_cols].isnull().sum()/df.shape[0]).map(lambda x: "{0:.2f}%".format(x * 100))
    num_col_summary['N_unique'] = df[num_cols].nunique()
    num_col_summary.sort_values(by = 'N_unique')
    return num_col_summary

def describe_cat_col(df, cat_cols):
    '''
    Get descriptive stats of categorical columns - quantiles, missing rate, number of unique values
    '''
    cat_col_summary = df[cat_cols].describe().transpose()
    cat_col_summary['N_missing_perc'] = (df[cat_cols].isnull().sum()/df.shape[0]).map(lambda x: "{0:.2f}%".format(x * 100))
    cat_col_summary['N_unique'] = df[cat_cols].nunique()
    cat_col_summary.sort_values(by = 'N_unique')
    return cat_col_summary

####################  Checks before merging data  ###################

def match_id(df1, df2, id_col1, id_col2):
    """
    Compare the ID columns of two dataframe, return message of match or not
    Example:
        utils_eda.match_id(client_risk, client_feats, 'UID', 'UID')
        two dataframes have the same ID
    """
    
    df1_UID = set(df1[id_col1])
    df2_UID = set(df2[id_col2])
    if np.array_equal(df1_UID, df2_UID):
        print('two dataframes have the same set of ID')
    else:
        print('two dataframes have different sets of ID')

def unique_id(df, id_col):
    """
    Check if a column is the unique identifier of a dataframe, i.e. no duplication
    params: id_col can be a string (one column name) or a list (multiple column names)
    """
    if df[id_col].duplicated().sum() == 0:
        if isinstance(id_col, list):
            print(', '.join(id_col) + ' are the unique identifiers')
        else:
            print(id_col + ' is the unique identifier')
    else:
        if isinstance(id_col, list):
            print(', '.join(id_col) + ' are not the unique identifiers')
        else:
            print(id_col + ' is not the unique identifier') 
            
def data_clean_report(message, df, target_col):
    print(message + ', there are {} records with default rate {:.2f}%'.format(df.shape[0], df[target_col].mean() * 100))

def data_subset_report(message, df, target_col, mask):
    print(message + ': {} loans ({:.2f}% of records) default rate {:.2f}%'.format(df.loc[mask].shape[0], df.loc[mask].shape[0]/df.shape[0] * 100, df.loc[mask][target_col].mean() * 100))