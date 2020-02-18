import pandas as pd

# Rename the column:
# A column is a string to start with 'BLE', e.g. 'BLE-0010-45656',
# and the part behind 'BLE' is not fixed.
# Rename all this kind of column as 'BLE'

df = pd.DataFrame({'Name': ['Ariel', 'Bieber', 'Cao', 'Diaosi'] , 'A':[1,2,3,9], 'B':[3.3, 4.9, 1.0, 7.7], 'C':['good', 'bad', 'bad', 'good'], 'BLE-978-46767789': [65, 657, 78, 123]})

# df.columns.str.startswith('BLE') # It returns to an array of Boolean

blu_col = df.columns[df.columns.str.startswith('BLE')].tolist() # find the column of bluetooth. The column name may change of different device board
blu_col # A list that the columns starting with 'BLE'

df.rename(columns={y:'bluetooth'+str(x) for x,y in enumerate(blu_col)}, inplace=True) # in case that there are multiple BLE columns

train = df.drop(['id', 'loss'], axis=1)
all_features = [x for x in train.columns]
cat_features = [x for x in train.select_dtypes(include=['object']).columns] # output: ['C', 'Name']
num_features = [x for x in train.select_dtypes(exclude=['object']).columns]

df['C'] = df['C'].astype('category').cat.codes # output: a series with value: 1, 0, 0, 1 

# dataframe weighted average:
# groupby之后，求每个group里面，根据某一列作为weights，其他几列的加权平均值
# https://stackoverflow.com/a/33575217
def weighted_sum(df, var_list, weight_var):
    """
    output weighted average for the columns with another column as weights.
    `np.average(df[var_list], axis=0, weights=df[weight_var]` will calucate
    the averages of columns `var_list`, with weights of `weight_var`;
    Then make the output array as a pandas series.
    """
    return pd.Series(np.average(df[var_list], axis=0, weights=df[weight_var]), index=var_list)
df.groupby(by='C').apply(weighted_sum, var_list=['A'], weight_var='B')
