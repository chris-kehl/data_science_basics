#using the bank train data

import numpy as np
import matplotlib.pyplot as plt

# we are going to save two variables seperately bt_age_y and bt_age_n
bt_age_y = bank_train[bank_train.response == "yes"]['age']
bt_age_n = bank_train[bank_train.response == "no"]['age']

# creating a stacked histogram of the two vaiables created above
plt.hist([bt_age_y, bt_age_n], bins = 10, stacked = True)
plt.legend(['Response = Yes', 'Response = No'])
plt.title('Histogram of Age with Response Overlay')
plt.xlable('Age'); plt.ylabel('Frequency'); plt.show()

# create a normalized histogram
# prep the data

# n is the height of the histogram bars, bins are the boundaries 
(n, bins, patches) = plt.hist([bt_age_y, bt_age_n], bins = 10, stacked = True)

# matrix construction to combine the heights of the two variable bars into one array 
# using the column_stack command
n_table = np.column_stack((n[0], n[1]))

# calculate what portion of the bar is accounted for by each variable
n_norm = n_table / n_table.sum(axis=1)[:, None]

# final preparation create an array whose rows are the exact cuts of each bin.
ourbins = np.column_stack((bins[0:10], bins[1:11]))

p1 = plt.bar(x = ourbins[:,0], height = n_norm[:,0], width = ourbins[:, 1] - ourbins[:, 0])
p2 = plt.bar(x = ourbins[:,0], height = n_norm[:,1], width = ourbins[:,1] - ourbins[:, 0], bottom = n_norm[:,0])
plt.legend(['Response = Yes', 'Response = No'])
plt.title('Normalize Histogram of Age with Response Overlay')
plt.xlabel('Age'); plt.ylabel('Proportion');plt.show()
