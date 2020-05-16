#Target definition
#Dataset has a target that could be default in X months
def target_creation(df):
    print('---- Target Creation started ----')
    df['target'] = (df['Loan_status']=="Default").astype(int)
    print('Number of bads is : ', df['target'].sum())
    print('Number of goods is : ', (df['target']==0).sum())
    print('bad rate is : {:.2%}'.format(df['target'].mean()))
    print('---- Target Creation completed ----')
    print('\n')
    return df