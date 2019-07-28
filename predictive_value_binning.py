# import pandas 
import pandas as pd

# cut() bins the values from the pandas package

bank_train['age_binned'] = pd.cut(x = bank_train['age'], bins = [0, 27, 60.01, 100],labels=["Under 27", "27 to 60", "Over 60"], right = False)

# Results is a bar graph with age binned and a response overlay
crosstab_02 = pd.crosstab(bank_train['age_binned'], bank_train['response'])
crosstab_02.plot(kind='bar', stacked = True, title = 'Bar Graph of Age (Binned) with Response Overlay')

# create a normalized bar graph
crosstab_norm = crosstab_02.div(crosstab_02.sum(1), axis = 0)
crosstab_norm.plot(kind='bar', stacked = True)

