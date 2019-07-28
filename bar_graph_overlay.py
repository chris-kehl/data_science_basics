# import Pandas
import pandas as pd

# import the datasets we will be working with
bank_train = pd.read_csv("~/Documents/data_files/data_science/data_sets/website_data_sets/bank_marketing_training")

# create a contingency table of the predictor and target variables
crosstab_01 = pd.crosstab(bank_train['previous_outcome'], bank_train['response'])

# create a bar graph based on the table
crosstab_01.plot(kind='bar', stacked = True)

# create a normalized version of the data
# change table so the values are the proportions to yes and no

crosstab_norm = crosstab_01.div(crosstab_01.sum(1), axis = 0)
 
# visualize the crosstab_norm by using the code for a stacked bar chart
crosstab_norm.plot(kind='bar', stacked = True)

# How to Construct Contingency Tables Using Python
crosstab_02 = pd.crosstab(bank_train['response'], bank_train['previous_outcome'])

round(crosstab_02.div(crosstab_02.sum(0), axis =1)*100, 1)
