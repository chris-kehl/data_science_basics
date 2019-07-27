# import Pandas
import pandas as pd

# import the datasets we will be working with
bank_train = pd.read_csv("/Users/chriskehl/Library/CloudStorage/iCloud Drive/Documents/data_files/data_science/data_sets/website_data_sets/bank_marketing_training")

# find the number of rows and columns in the dataset by using .shape
bank_train.shape

# we are adding an index column  Note: python starts at 0 not 1
bank_train['index'] = pd.Series(range(0,26874))

# view the head of the file
bank_train.head