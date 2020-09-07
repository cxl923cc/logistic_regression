import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import re
def split_num_cat(s, special_values = []):
    '''
    Take a pandas series (s) as input
    Return two series 
    - 's_num' containing the numeric part
    - 's_cat' containing the categorical part, including missing value and special values (customised list)
    Example:
        s = pd.Series([1, 2, 3, np.nan, 'a', 'b'])
        s_num, s_cat = split_num_cat(s, special_values = [3])
        print(s_num, s_cat)
        0    1.0
        1    2.0
        dtype: float64 
        2      3
        3    NaN
        4      a
        5      b
        dtype: object
    '''
    #Create mask to treat missing and special values as categorical
    mask_not_null_or_special = s.notnull() & ~s.isin(special_values)
    
    #Keep numeric
    s_num = pd.to_numeric(s.loc[mask_not_null_or_special], errors = 'coerce').dropna()
    
    #Take the elements that have indices in s but not s_num
    s_cat = s.loc[s.index.difference(s_num.index)].astype(int)
    
    s_cat = s_cat.apply(str)
    
    return s_num, s_cat

# Fineclass for s_num   s_num -> <num_fineclass> -> s_num_fc and num_bin_edges_fc
def num_fc(s_num, bin_num_fineclass = 20):
    '''
    params:
    - bin_num_fineclass: number of fineclass bins, by default 20. at least 2.
    Return the binned series and a numpy array of the bin edges
    Example:
        s_num_fc, num_bin_edges_fc = num_fc(df['Borrower_rate'])
        num_bin_edges_fc
        
        array([0.0274, 0.0294, 0.0314, 0.0333, 0.0334, 0.0381, 0.0496, 0.0582,
               0.0724, 0.0754, 0.092 , 0.116 , 0.133 , 0.149 , 0.168 , 0.2   ,
               0.243 , 0.32  ])
    '''
    s_num_fc = pd.Series(name = s_num.name)
    num_bin_edges_fc = []
    
    if len(s_num)>0:
        s_interval = pd.qcut(s_num, q=bin_num_fineclass, duplicates = 'drop')
        num_bins = s_interval.value_counts().shape[0] #sometimes there is a bin with 0 record, have to use value_counts to see it instead of unique
        print('num of bins is ',num_bins)
        s_interval_bin_number  = pd.qcut(s_num, q=bin_num_fineclass, duplicates = 'drop', labels = range(num_bins)).astype(str)

        #Add number in front of the interval, convert interval to string for better compatibility with matplotlib in later functions
        s_label = s_interval.map(lambda x: str(round(x.left, 4)) + '< - <=' + str(round(x.right,4))).astype('str')
        s_num_fc = s_interval_bin_number.str.cat(s_label, sep='. ')

        unique_intervals = s_interval.unique().sort_values()
        num_bin_edges_fc = np.append(unique_intervals.map(lambda x: x.left).astype(float).min(), unique_intervals.map(lambda x: x.right).astype(float))
        
        #avoid min or max missed out from data due to truncation to 4 digit float format
        num_bin_edges_fc[0] = np.minimum(np.min(s_num), num_bin_edges_fc[0])
        num_bin_edges_fc[-1] = np.maximum(np.max(s_num), num_bin_edges_fc[-1])
        
    return s_num_fc, num_bin_edges_fc

# Fineclass for s_cat   s_cat -> <cat_fineclass> -> s_cat_cc and cat_bin_values_fc
def cat_fc(s_cat, missing_label = 'Missing', special_value_label_dict = {}):
    '''
    fineclass for categorical series
    Return the binned series and a numpy array of the bin values
    Example:
        s_cat_fc, cat_bin_values_fc = cat_fc(s_cat=(df['Days_in_arrears']))
        cat_bin_values_fc
        
        array(['Missing', '1-30 days', '>90 days', '61-90 days', '31-60 days'],
              dtype=object)
    '''
    s_cat_fc = s_cat.fillna(missing_label)
    # Replace special values with the labels     
    s_cat_fc = s_cat_fc.replace(special_value_label_dict)
    
    cat_bin_values_fc = np.sort(s_cat_fc.unique())
    
    return s_cat_fc, cat_bin_values_fc

# Stack s_num_fc and s_cat_fc
def combine_num_cat_fc(s_num_fc, s_cat_fc):
    s_fc = s_num_fc.append(s_cat_fc)
    return s_fc
    
# WOE IV Stats
def bin_woe_iv(s, target, special_values_dict, desc = 'Fineclass'):
    '''
    Pass the feature and target series in (both should have the same index)
    Return the dataframe of woe IV stats and plots
    params:
        desc: by default 'Fineclass', can set as Coarse class with customised description
    '''
    eval_df = pd.DataFrame({'feat': s, 'target': target})
    eval_df.sort_values(by='feat')
    eval_df['good'] = (eval_df['target']==0)
    eval_df['bad'] = (eval_df['target']==1)
    eval_df_summary = eval_df.groupby('feat')[['good', 'bad']].sum().rename(columns = {'good':'N_good', 'bad':'N_bad'}).reset_index()
    #get the numeric order of num part so 10. is not ranked before 2.
    eval_df_summary['feat_num_index'] = pd.to_numeric(eval_df_summary['feat'].map(lambda x: x.split('.')[0]),errors='coerce')
    if s.name in special_values_dict.keys():
        eval_df_summary.loc[eval_df_summary['feat'].isin(list(map(str, special_values_dict[s.name]))), 'feat_num_index'] = np.nan
    eval_df_summary.sort_values(by=['feat_num_index', 'feat'], inplace = True)
    eval_df_summary = eval_df_summary.reset_index(drop=True)
    eval_df_summary['N_count'] = eval_df_summary['N_good'] + eval_df_summary['N_bad']
    num_good_total = eval_df_summary['N_good'].sum()
    num_bad_total = eval_df_summary['N_bad'].sum()

    eval_df_summary['dist_good']=eval_df_summary['N_good']/num_good_total
    eval_df_summary['dist_bad']=eval_df_summary['N_bad']/num_bad_total
    eval_df_summary['bin_count_perc'] = eval_df_summary['N_count']/(num_good_total + num_bad_total)
    eval_df_summary['woe'] = np.log(eval_df_summary['dist_good']/eval_df_summary['dist_bad']).replace(np.inf, np.nan)
    eval_df_summary['logodds'] = np.log(eval_df_summary['N_good']/eval_df_summary['N_bad']).replace(np.inf, np.nan)
    eval_df_summary['p_bad'] = (eval_df_summary['N_bad']/eval_df_summary['N_count']).replace(np.inf, np.nan)
    eval_df_summary['iv'] = (eval_df_summary['dist_good']-eval_df_summary['dist_bad'])*eval_df_summary['woe']
    
    s_total = eval_df_summary.sum().drop(['feat_num_index', 'woe', 'logodds', 'p_bad'])
    s_total.iloc[0] = 'Total'
    
    eval_df_summary = eval_df_summary.append(s_total, ignore_index = True)
    eval_df_summary['var'] = s.name
    eval_df_summary['desc'] = desc
    
    print(desc)
    print('IV is ' + str(eval_df_summary['iv'].sum()))
#     if eval_df_summary['N_count'].sum()==((target==0)|(target==1)).sum():
#         print('total number of rows match')
#     else:
#         print('total number of rows DOES NOT match')
    
    eval_df_summary_forplot = eval_df_summary.loc[eval_df_summary['feat']!='Total']
    plt.plot(eval_df_summary_forplot['woe'], label='woe', marker = '.')
    plt.title(s.name + ' woe')
    plt.xticks(ticks = eval_df_summary_forplot.index.values, labels=eval_df_summary_forplot['feat'].values)
    plt.xticks(rotation=60)
    plt.legend(loc='upper left')
    plt.twinx()
    plt.bar(eval_df_summary_forplot.index, eval_df_summary_forplot['bin_count_perc'], alpha=0.1, label='bin_vol%')
    plt.legend(loc='upper right')
    plt.show()
    return eval_df_summary
    
# Define coarseclass for s_num  1) s_num and num_bin_edges_fc -> <zero bad treatment> -> num_bin_edges_cc -> <customised bin edges optimisation> -> num_bin_edges_cc 
def num_cc_edges_nonzerobad(s_num, num_bin_edges_fc, eval_df_summary_fc):
    '''
    if there is any bin with 0 number of bads, combine it with the next non-zero bad bin
    Return a numpy array of the bin edges
    params: 
        num_bin_edges_fc: the pandas series of target
        eval_df_summary_fc: the output dataframe of 20 bin fineclass
    Example:
        num_bin_edges_cc = num_cc_edges_nonzerobad(s_num, num_bin_edges_fc, eval_df_summary_fc)
        num_bin_edges_cc
        
        array([0.0274, 0.0724, 0.092 , 0.133 , 0.149 , 0.168 , 0.2   , 0.243 ,
       0.32  ])
    '''
    # for the non 0 N_bad bins for numeric bins, get an array of the max in the intervals (min <- <= max) 
    nonzero_bad_bin_edge_max = pd.to_numeric(eval_df_summary_fc.loc[(eval_df_summary_fc['N_bad']!=0) & (eval_df_summary_fc['feat_num_index'].notnull())]['feat']\
                                                                               .map(lambda x: x.split('<=')[-1])).dropna().astype(float)
    # add min in front of it to make it friendly for pd.qcut
    num_bin_edges_cc = np.append(num_bin_edges_fc.min(), nonzero_bad_bin_edge_max)
    # In case the highest value bins have 0 bad, add the max to the end to make it friendly for pd.qcut
    if num_bin_edges_cc.max()<num_bin_edges_fc.max():
        num_bin_edges_cc = np.append(num_bin_edges_cc[:-1], num_bin_edges_fc.max())

    #avoid min or max missed out from data due to truncation to 4 digit float format
    num_bin_edges_cc[0] = np.minimum(np.min(s_num), num_bin_edges_cc[0])
    num_bin_edges_cc[-1] = np.maximum(np.max(s_num), num_bin_edges_cc[-1])
    
    return num_bin_edges_cc


def num_cc_edges_woe_enforce_monotonicity(s_num, target, num_bin_edges, eval_df_summary, special_values_dict):
    #Remove special value row from the summary data
    eval_df_summary = eval_df_summary.loc[~eval_df_summary['feat'].isin(list(map(str, special_values_dict[s_num.name])))]
    
    #Get the delta woe for ith row as woe of ith row - woe of (i+1)th row
    eval_df_summary['woe_diff'] = eval_df_summary['woe'].diff(-1)
    eval_df_summary['woe_diff_abs_forward'] = eval_df_summary['woe_diff'].abs()
    eval_df_summary['woe_diff_abs_backward'] = eval_df_summary['woe_diff_abs_forward'].shift(1)
    eval_df_summary['bin_edge_min'] = eval_df_summary['feat'].map(lambda x: re.split(' |<', x)[1]).astype(float)
    eval_df_summary['bin_edge_max'] = eval_df_summary['feat'].map(lambda x: x.split('<=')[-1]).astype(float)
    # if the correlation is positive, woe should decrease as feature increase, detect negative woe_diff; vice versa
    feat_corr = s_num.corr(target.loc[s_num.index])
    if s_num.corr(target.loc[s_num.index])>0:
        #positive woe_diff correspond bins that have higher woe than the next bin
        print('feature and target correlation is {}, woe should decrease as feature increase'.format(feat_corr))
        non_monoto_bins = eval_df_summary.loc[eval_df_summary['woe_diff']<=0]
    else:
        print('feature and target correlation is {}, woe should increase as feature increase'.format(feat_corr))
        non_monoto_bins = eval_df_summary.loc[eval_df_summary['woe_diff']>=0]

    #comparing the woe_diff_abs of group backwards or forwards to drop min or max of the bin edges
    #if by default group forward (drop max), but if forward is missing (last row), or woe diff abs backward < forward, group backward (drop min)
    non_monoto_bins['bin_edges_index_to_drop'] = np.where(((non_monoto_bins['woe_diff_abs_forward'].isnull()) | \
                                                     (non_monoto_bins['woe_diff_abs_backward']<non_monoto_bins['woe_diff_abs_forward'])),\
                                                      non_monoto_bins.index,
                                                      non_monoto_bins.index+1)
    
    print('input bin edges : ', num_bin_edges, '\n', 'dropping index', non_monoto_bins['bin_edges_index_to_drop'].values)
        
    num_bin_edges_cc = np.delete(num_bin_edges, non_monoto_bins['bin_edges_index_to_drop'])
#     np.array([x for x in num_bin_edges if round(x,4) not in non_monoto_bins['bin_edges_index_to_drop'].values])

    #avoid min or max missed out from data due to truncation to 4 digit float format
    num_bin_edges_cc[0] = np.minimum(np.min(s_num), num_bin_edges_cc[0])
    num_bin_edges_cc[-1] = np.maximum(np.max(s_num), num_bin_edges_cc[-1])
    
    print('output bin edges : ', num_bin_edges_cc)
    
    return num_bin_edges_cc


def num_cc_edges_custom(num_bin_edges, num_bin_edges_groups):
    '''
    Input a list of lists of group number starting from 0
    Example:
        #to group 1 and 2 together, 3 and 4 together
        num_cc_edges_custom(num_bin_edges_cc, [[3,4]])
        input bin edges :  [-0.0010 0.9080 1.6360 2.1320 3.3720 4.5200 6.6490 32.7040] 
         dropping  [3.372]
        output bin edges :  [-0.0010 0.9080 1.6360 2.1320 4.5200 6.6490 32.7040]
    '''
    bin_edges_to_drop =[]
    for group in num_bin_edges_groups:
        assert len(group)==2, 'group two bins at a time for best outcome'
        #drop the max edge of group[0] based on index
        bin_edges_to_drop.append(num_bin_edges[group[0]+1])
        
    print('input bin edges : ', num_bin_edges, '\n', 'dropping ', bin_edges_to_drop)
        
    num_bin_edges_cc = np.array([x for x in num_bin_edges if x not in bin_edges_to_drop])
    
    num_bin_edges_cc = num_bin_edges_cc
    
    print('output bin edges : ', num_bin_edges_cc)    
    return num_bin_edges_cc

def num_cc(s_num, num_bin_edges_cc):
    '''
    Input coarse class bin edges
    Return a numpy array of the bin edges
    Example:
        num_bin_edges_cc = num_cc_edges_nonzerobad(s_num, num_bin_edges_fc, eval_df_summary_fc)
        s_num_cc = num_cc(s_num, num_bin_edges_cc)
        s_num_cc
        
        0      0. 0.0274< - <=0.0724
        1       1. 0.0724< - <=0.092
        2      0. 0.0274< - <=0.0724
        3      0. 0.0274< - <=0.0724
        4      0. 0.0274< - <=0.0724
                       ...          
        901        6. 0.2< - <=0.243
        902        6. 0.2< - <=0.243
        903    0. 0.0274< - <=0.0724
        904        5. 0.168< - <=0.2
        905        6. 0.2< - <=0.243
    '''

    s_interval = pd.cut(s_num, bins=num_bin_edges_cc, duplicates = 'drop')
    num_bins = s_interval.value_counts().shape[0] #sometimes there is a bin with 0 record, have to use value_counts to see it instead of unique
    s_interval_bin_number  = pd.cut(s_num, bins=num_bin_edges_cc, duplicates = 'drop', labels = range(num_bins)).map(lambda x: str(int(x)))

    #Add number in front of the interval, convert interval to string for better compatibility with matplotlib in later functions
    s_label = s_interval.map(lambda x: str(x.left) + '< - <=' + str(x.right)).astype('str')
    s_num_cc = s_interval_bin_number.str.cat(s_label, sep='. ')
    
    return s_num_cc

def cat_cc(s_cat, cat_bin_values, cat_bin_values_groups):
    '''
    Input a list of lists of group number starting from 0
    Return series with bin labels replaced by group bin labels, and an numpy array of grouped bin labels 
    Example:
        #to group 0,1,2 together; 3, 4 together; 5, 6 together
        s_cat_cc, cat_bin_values_cc = cat_cc(s_cat, cat_bin_values = cat_bin_values_fc, cat_bin_values_groups = [[0, 1, 2], [3, 4], [5, 6]])
        input bin groups [[0, 1, 2], [3, 4], [5, 6]]
        output bin values  ['A, A*, A-Sec' 'B, C' 'D, E']
    '''
    s_cat_cc = s_cat
    print('input bin groups ', cat_bin_values_groups)
    for group in cat_bin_values_groups:
        assert len(group)>=2, 'group at least two bins at a time'
        
        s_cat_cc[np.isin(s_cat_cc, cat_bin_values[group])] = ', '.join(cat_bin_values[group])
        
    cat_bin_values_cc = np.sort(s_cat_cc.unique())
    print('output bin values ', cat_bin_values_cc)
    
    return s_cat_cc, cat_bin_values_cc

def cc_cat_zerobad_lowvolwoesmooth(cat_bin_values, eval_df_summary, min_bin_count=20, smoothing=10, min_n_bad = 0.01, desc = '1. Coarseclass smoothe 0 bad bin woe'):
    '''
    Example:
        eval_df_summary_cc = cc_cat_zerobad_lowvolwoesmooth(cat_bin_values = cat_bin_values_fc, \
        eval_df_summary = eval_df_summary_fc, min_bin_count=20, smoothing=10, min_n_bad = 0.01)
        eval_df_summary_cc
    '''
    #Assign 0.1 bad to zero bad bin
    eval_df_summary.loc[((eval_df_summary['N_bad']==0)), 'N_bad'] = min_n_bad
    
    #Recalculate woe iv
    eval_df_summary['N_count'] = eval_df_summary['N_good'] + eval_df_summary['N_bad']
    num_good_total = eval_df_summary['N_good'].sum()
    num_bad_total = eval_df_summary['N_bad'].sum()

    eval_df_summary['dist_good']=eval_df_summary['N_good']/num_good_total
    eval_df_summary['dist_bad']=eval_df_summary['N_bad']/num_bad_total
    eval_df_summary['bin_count_perc'] = eval_df_summary['N_count']/(num_good_total + num_bad_total)
    eval_df_summary['woe'] = np.log(eval_df_summary['dist_good']/eval_df_summary['dist_bad']).replace(np.inf, np.nan)
    
    #Smooth woe for low volume bins (at least 20 observations to be meaningful)
    eval_df_summary['smoothing_factor'] =   1 / (1 + np.exp(-(eval_df_summary['N_count'] - min_bin_count) / smoothing)) 
    eval_df_summary.loc[eval_df_summary['N_bad']==min_n_bad, 'woe'] = eval_df_summary.loc[eval_df_summary['N_bad']==min_n_bad, 'smoothing_factor'] * eval_df_summary.loc[eval_df_summary['N_bad']==min_n_bad, 'woe']
    
    eval_df_summary['logodds'] = np.log(eval_df_summary['N_good']/eval_df_summary['N_bad']).replace(np.inf, np.nan)
    eval_df_summary['p_bad'] = (eval_df_summary['N_bad']/eval_df_summary['N_count']).replace(np.inf, np.nan)
    eval_df_summary['iv'] = (eval_df_summary['dist_good']-eval_df_summary['dist_bad'])*eval_df_summary['woe']
    
    var_name = eval_df_summary['var'].unique()[0]
    s_total = eval_df_summary.sum().drop(['feat_num_index', 'woe', 'logodds', 'p_bad', 'var'])
    s_total.iloc[0] = 'Total'
    
    eval_df_summary = eval_df_summary.append(s_total, ignore_index = True)
    
    eval_df_summary['var'] = var_name
    
    eval_df_summary['desc'] = desc
    
    return eval_df_summary

def cc_to_woe(s_cc, eval_df_summary_cc):
    '''
    Input post coarse classing series and the woe summary data frame
    Return the woe series
    Example:
        s_cc_woe = cc_to_woe(s_cc, eval_df_summary_cc)
        s_cc_woe
        
        {'0. 0.0274< - <=0.0724': 1.7180816231078904, '1. 0.0724< - <=0.133': 0.32514271926933125, 
        '2. 0.133< - <=0.168': -0.591148012604824, '3. 0.168< - <=0.2': -0.6601408840917752, 
        '4. 0.2< - <=0.243': -1.3532880646517205, '5. 0.243< - <=0.32': -1.7587531727598849}
        0      1.718082
        1      0.325143
        2      1.718082
        3      1.718082
        4      1.718082
                 ...   
        901   -1.353288
        902   -1.353288
        903    1.718082
        904   -0.660141
        905   -1.353288
        Name: Borrower_rate_woe, Length: 906, dtype: float64
    '''
    woe_dict = dict(zip(eval_df_summary_cc['feat'], eval_df_summary_cc['woe']))
    print('woe encoding', woe_dict)
    s_cc_woe = s_cc.replace(woe_dict)
    s_cc_woe.name = s_cc.name + '_woe'
    return s_cc_woe, woe_dict