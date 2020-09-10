# Low level modules used for binning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Low-level functions for numeric features
# 1.make fineclass bins (fineclass)
# 2.make custom bin (general)
# 3.get bin label (general)
# 4.build attribute level woe and iv report (general)
# 5.get woe feature (general)
# 6.upsert_fineclass_summary (general)

# Medium-level functions
# 1.num_fineclass that uses 1, 3, 4, 5
# 2.num_custom_bin that uses 2, 3, 4, 5
# 3.cat_fineclass that uses 4, 5

#################    Low Level Functions ########################

def num_make_fc_bin(raw_feat, num_bin = 20):
    '''
    Given number of bins, equally divide the whole range of raw values of the feature into equal sized bins, 
    return a Series of bin numbers (1, 2, 3, etc.), and the bin cut offs
    
    params:
        raw_feat: the original feature before binning
        num_bin: the number of equal sized bins. Note that bins won't always be of equal size. It depends on distribution.
    '''
    
    feat_fc, bin_cut_offs_fc = pd.qcut(raw_feat, q = num_bin, duplicates = 'drop', labels=False, retbins=True)
    
    return feat_fc, bin_cut_offs_fc

def num_make_custom_bin(raw_feat, bin_cut_offs):
    ''' Given raw feature and a list of bin cut offs, return a Series of bin number
    params:
        raw_feat: the raw feature
        bin_cut_offs: a list of bin cut offs. can be the output of fineclass binning (num_make_bin) or a user-defined list.
    '''
    binned_feat = pd.cut(raw_feat, bins = bin_cut_offs, labels=False, include_lowest = True)
    assert binned_feat.isnull().sum()==0, '{}: Check the values in the input Series. Some value is outside of the range of bin cut offs, \
                                            and it is not mapped to a bin'.format(raw_feat.name)
    return binned_feat

def num_get_bin_label(bin_cut_offs):
    '''
    Given bin cut offs, create bin label, return a dict of bin cutoffs to bin label for better readbility of the binning report
    params:
        bin_cut_offs: a list of bin cutoffs
    
    example:
        array([ 0.,  1.,  2., 19.]) -> {0: '1. [0.0, 1.0]', 1: '2. (1.0, 2.0]', 2: '3. (2.0, 19.0]'}
    '''
    
    bin_label_dict = {}
    
    for i, cutoff in enumerate(bin_cut_offs[:-1], 1):
        fmt_str = '{}. [{}, {}]' if i == 1 else '{}. ({}, {}]'
        bin_label_dict[i-1] = fmt_str.format(str(i), cutoff, bin_cut_offs[i])
    
    return bin_label_dict

def calc_woe_iv(target, feat_input, bin_label_dict, feat_col = 'CNT_CHILDREN', desc = '0.Fineclass', show_plot = True):
    '''
    Calculate attribute level woe and iv, return a dataframe of attr_woe_iv that is a report of the binned feature
    
    params:
        target: the target Series
        feat_input: the feature Series. could be the output of fineclass binning (num_make_fc_bin) or coarseclass binning
        bin_label_dict: map bin number to bin label for better readbility of the bins
        feat_col: pass the name of the feature to show in the attribute level woe table and plot
        desc: provide description for logging every step, i.e. 0.Fineclass, 1.Coarseclass
        show_plot: True by default. If False, the plot of attribute level bad rate and volume will not be shown.
    '''

    eval_df = pd.DataFrame({'feat': feat_input, 'target': target})
    eval_df.sort_values(by='feat')
    assert (~eval_df['target'].isin([0, 1])).sum()==0, 'target should only have two values - either good or bad'
    eval_df['good'] = (eval_df['target']==0)
    eval_df['bad'] = (eval_df['target']==1)

    # count good and bad for each bin
    attr_woe_iv = eval_df.groupby('feat')[['good', 'bad']].sum().rename(columns = {'good':'N_good', 'bad':'N_bad'}).reset_index()
    attr_woe_iv['N_count'] = attr_woe_iv['N_good'] + attr_woe_iv['N_bad']
    num_good_total = attr_woe_iv['N_good'].sum()
    num_bad_total = attr_woe_iv['N_bad'].sum()

    # calculate woe, logodds, iv
    attr_woe_iv['dist_good']=attr_woe_iv['N_good']/num_good_total
    attr_woe_iv['dist_bad']=attr_woe_iv['N_bad']/num_bad_total
    attr_woe_iv['bin_count_perc'] = attr_woe_iv['N_count']/(num_good_total + num_bad_total)
    attr_woe_iv['woe'] = np.log(attr_woe_iv['dist_good']/attr_woe_iv['dist_bad']).replace(np.inf, np.nan)
    attr_woe_iv['logodds'] = np.log(attr_woe_iv['N_good']/attr_woe_iv['N_bad']).replace(np.inf, np.nan)
    attr_woe_iv['p_bad'] = (attr_woe_iv['N_bad']/attr_woe_iv['N_count']).replace(np.inf, np.nan)
    attr_woe_iv['iv'] = (attr_woe_iv['dist_good']-attr_woe_iv['dist_bad'])*attr_woe_iv['woe']

    attr_woe_iv['var'] = feat_col
    attr_woe_iv['desc'] = desc
    attr_woe_iv['bin_label'] = attr_woe_iv['feat'].map(bin_label_dict)


    if show_plot:
        plt.plot(attr_woe_iv['woe'], label='woe', marker = '.')
        plt.title(feat_col + ' iv: {:.4f}'.format(attr_woe_iv['iv'].sum()))
        plt.xticks(ticks = attr_woe_iv.index.values, labels=attr_woe_iv['bin_label'].values)
        plt.xticks(rotation=60)
        plt.legend(loc='upper left')
        plt.twinx()
        plt.bar(attr_woe_iv.index, attr_woe_iv['bin_count_perc'], alpha=0.1, label='bin_vol%')
        plt.legend(loc='upper right')
        plt.show()

    return attr_woe_iv

def get_woe_feat(feat_input, attr_woe_iv):
    ''' Given the feature and attribute level woe dataframe, return the woe feature
    params:
        feat_input: the binned feature can be the output of fineclass binning (num_make_bin) or coarse class binning, or even the original feature.
        attr_woe_iv: the attribute level woe dataframe for creating the dictionary to map bins to woe
    '''
    woe_dict = dict(zip(attr_woe_iv['feat'], attr_woe_iv['woe']))
    woe_feat = feat_input.map(woe_dict)
    woe_feat.name = woe_feat.name + '_woe'
    if woe_feat.isnull().sum()==0:
        warnings.warn('{}: Some bins are missing or not matched to woe. Check if there is missing value in the binned feature \
                                        or the binned feature does not match the values in woe dataframe.'.format(woe_feat.name))
        
    return woe_feat

def upsert_fineclass_summary(fineclass_summary, feat_col, attr_woe_iv, desc):
    #Remove existing log in the report if there is any
    fineclass_summary = fineclass_summary.loc[~((fineclass_summary['var']==feat_col) & (fineclass_summary['desc'] == desc))]
    #Append new log to the report     
    fineclass_summary = pd.concat([fineclass_summary, attr_woe_iv], axis = 0)
    return fineclass_summary
    
#################    Medium Level Function ########################

def num_fineclass(df, target, feat_col, num_bin = 20, desc = '0.Fineclass'):
    ''' Given the name of the feature and number of bins to make, Return a Series of woe and a report about attribute level woe and iv
        The process is to get bin cut off -> make bin and get bin label -> get attribute level woe -> get woe feature
    params:
        df: the dataframe
        feat_col: the name of the feature column
        num_bin: the number of bins for groupping the raw feature into equally-sized bins. 20 by default.
        desc: the description of the binning step used in the report. '0.Fineclass' by default.
    '''
    raw_feat = df[feat_col]
    feat_fc, bin_cut_offs_fc = num_make_fc_bin(raw_feat, num_bin = num_bin)
    bin_label_dict = num_get_bin_label(bin_cut_offs = bin_cut_offs_fc)
    attr_woe_iv = calc_woe_iv(target, feat_input = feat_fc, bin_label_dict = bin_label_dict, \
                                       feat_col = feat_col, desc = desc)
    woe_feat = get_woe_feat(feat_input = feat_fc, attr_woe_iv = attr_woe_iv)
    return woe_feat, attr_woe_iv

def num_custom_bin(df, target, feat_col, bin_cut_offs, desc = '1.Custom'):
    ''' Given the name of the feature and customised cut offs, Return a Series of woe and a report about attribute level woe and iv
        The process is to apply bin cut off -> make bin and get bin label -> get attribute level woe -> get woe feature
    params:
        df: the dataframe
        feat_col: the name of the feature column
        bin_cut_offs: the user defined cut offs
        desc: the description of the binning step used in the report. '1.Custom' by default.
    '''
    raw_feat = df[feat_col]
    binned_feat = num_make_custom_bin(raw_feat, bin_cut_offs)
    bin_label_dict = num_get_bin_label(bin_cut_offs)
    attr_woe_iv = calc_woe_iv(target, feat_input = binned_feat, bin_label_dict = bin_label_dict, \
                                       feat_col = feat_col, desc = desc)
    woe_feat = get_woe_feat(feat_input = binned_feat, attr_woe_iv = attr_woe_iv)
    return woe_feat, attr_woe_iv

def cat_fineclass(df, target, feat_col, bin_label_dict = None, desc = '0.Fineclass'):
    ''' Given the name of the categorical feature, Return a Series of woe and a report about attribute level woe and iv
        Can also be used to bin the numeric features with 2 values.
        The process is to use each unique value as a bin, use value as bin label -> get attribute level woe -> get woe feature
    params:
        df: the dataframe
        feat_col: the name of the feature column
        bin_label_dict: Optional. A user defined dictionary to map raw value to bin label. by default it keeps the raw value.
        desc: the description of the binning step used in the report. '0.Fineclass' by default.
    '''
    raw_feat = df[feat_col]
    if bin_label_dict == None:
        bin_label_dict = dict(zip(raw_feat.unique(), raw_feat.unique()))
    attr_woe_iv = calc_woe_iv(target, feat_input = raw_feat, bin_label_dict = bin_label_dict, \
                                       feat_col = feat_col, desc = desc)
    woe_feat = get_woe_feat(feat_input = raw_feat, attr_woe_iv = attr_woe_iv)
    return woe_feat, attr_woe_iv