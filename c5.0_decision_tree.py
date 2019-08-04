
# import the required python packages

import pandas as pd
import numpy as np
import statsmodels.tools.tools as stattools
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# import the dataa
adult_tr = pd.read_csv("/Users/chriskehl/Library/CloudStorage/iCloud Drive/Documents/data_files/data_science/data_sets/website_data_sets/adult_ch6_training")

# save the income variable as y
y = adult_tr[['Income']]

#make dummy variables
mar_np = np.array(adult_tr['Marital status'])
(mar_cat, mar_cat_dict) = stattools.categorical (mar_np, drop=True, dictnames = True)

# add newly created dummy variables back into the x variables
mar_cat_pd = pd.DataFrame(mar_cat)
X = pd.concat((adult_tr[['Cap_Gains_Losses']], mar_cat_pd), axis = 1)

# specify the column names in X
X_names = ["Cap_Gains_Losses", "Divorced", "Married", "Never-Married", "Separated", "Widowed"]

# explain the levels of Y
y_names = ["<=50K", ">50K"]

# build the decisionTree using entropy as a classifier
c50_01 = DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 5).fit(X,y)

# export the summary of the c5.0 decisionTree
export_graphviz(c50_01, out_file = "/Users/chriskehl/Library/CloudStorage/iCloud Drive/Documents/data_files/data_science/data_sets/website_data_sets/c50_01.dot", feature_names=X_names,class_names=y_names)

# run predict() with X as the input
c50_01.predict(X)
