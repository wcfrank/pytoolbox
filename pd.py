import pandas as pd

# Rename the column:
# A column is a string to start with 'BLE', e.g. 'BLE-0010-45656',
# and the part behind 'BLE' is not fixed.
# Rename all this kind of column as 'BLE'

df = pd.DataFrame({'Name': ['Ariel', 'Bieber', 'Cao', 'Diaosi'] , 'A':[1,2,3,9], 'B':[3.3, 4.9, 1.0, 7.7], 'BLE-978-46767789': [65, 657, 78, 123]})

# df.columns.str.startswith('BLE') # It returns to an array of Boolean

blu_col = df.columns[df.columns.str.startswith('BLE')].tolist() # find the column of bluetooth. The column name may change of different device board
blu_col # A list that the columns starting with 'BLE'

df.rename(columns={y:'bluetooth'+str(x) for x,y in enumerate(blu_col)}, inplace=True) # in case that there are multiple BLE columns