# Low level modules used for binning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

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
        if i == 1:
            bin_label_dict[i-1] = '{}. [{}, {}]'.format(str(i), cutoff, bin_cut_offs[i])
        else:
            bin_label_dict[i-1] = '{}. ({}, {}]'.format(str(i), cutoff, bin_cut_offs[i])
    
    return bin_label_dict


def calc_woe_iv(target, feat_input, bin_label_dict, feat_col = 'CNT_CHILDREN', desc = '0.Fineclass', show_plot = True):
    '''
    Calculate attribute level woe and iv, and feature level iv, return two dataframes - attr_woe_iv and feat_iv
    
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
    attr_woe_iv['bin_label'] = attr_woe_iv['feat'].map(bin_label_dict)
    attr_woe_iv['desc'] = desc

    feat_iv = pd.DataFrame({'var' : [feat_col],
                            'desc' : [desc],
                            'iv' : [attr_woe_iv['iv'].sum()]})

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

    return attr_woe_iv, feat_iv