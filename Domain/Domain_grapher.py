import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


""" All DATA ANALYSIS """
# Load Data
data = pd.read_csv("all_data.csv")
data_extracted = pd.DataFrame(data, columns=['num_max','denom_max','denom_min','epoch']).to_numpy()

# Data arrays
n_max = data_extracted[:,0]
d_max = data_extracted[:,1]
d_min = data_extracted[:,2]
epoch = data_extracted[:,3]

# Formatting Color Data for epoch goes from Red -> Purple over time
normalized_epoch = epoch/np.max(epoch)
color_epoch = []
for e in normalized_epoch:
    color_epoch.append((1-e,0,1))

""" Untruncated Plots for All Data """
# Histogram of max numerator values
plt.figure(1)
plt.hist(n_max, bins=1000)

# Histogram of maximum denominator values
plt.figure(2)
plt.hist(d_max, bins=10000)

# Scatter Plot of maximum denominator values
#plt.figure(3)
#plt.scatter(n_max,d_max,c=epoch,marker='.',cmap='winter')

# Histogram of minimum denominator values
plt.figure(4)
plt.hist(np.log(d_min), bins=1000)

# Scatter Plot of minimum denominator values
#plt.figure(5)
#plt.scatter(n_max,np.log(d_min),c=epoch,marker='.',cmap='autumn')

#plt.show()


""" RANGE 3 ANALYSIS """

# Load Data
data = pd.read_csv("range_file_3.csv")
data_extracted = pd.DataFrame(data, columns=['num_max','denom_max','denom_min','epoch']).to_numpy()

# Data arrays
n_max = data_extracted[:,0]
d_max = data_extracted[:,1]
d_min = data_extracted[:,2]
epoch = data_extracted[:,3]

# Formatting Color Data for epoch goes from Red -> Purple over time
normalized_epoch = epoch/np.max(epoch)
color_epoch = []
for e in normalized_epoch:
    color_epoch.append((1-e,0,1))

""" Untruncated Plots for Range 3 Data """
# Histogram of max numerator values
#plt.figure(6)
#plt.hist(n_max, bins=1000)

# Histogram of maximum denominator values
#plt.figure(7)
#plt.hist(d_max, bins=10000)

# Scatter Plot of maximum denominator values
#plt.figure(8)
#plt.scatter(n_max,d_max,c=epoch,marker='.',cmap='winter')

# Histogram of minimum denominator values
#plt.figure(9)
#plt.hist(np.log(d_min), bins=1000)

# Scatter Plot of minimum denominator values
#plt.figure(10)
#plt.scatter(n_max,np.log(d_min),c=epoch,marker='.',cmap='autumn')


""" RANGE 4 ANALYSIS """

# Load Data
data = pd.read_csv("range_file_4.csv")
data_extracted = pd.DataFrame(data, columns=['num_max','denom_max','denom_min','epoch']).to_numpy()

# Data arrays
n_max = data_extracted[:,0]
d_max = data_extracted[:,1]
d_min = data_extracted[:,2]
epoch = data_extracted[:,3]

# Formatting Color Data for epoch goes from Red -> Purple over time
normalized_epoch = epoch/np.max(epoch)
color_epoch = []
for e in normalized_epoch:
    color_epoch.append((1-e,0,1))

""" Untruncated Plots for Range 3 Data """
# Histogram of max numerator values
#plt.figure(6)
#plt.hist(n_max, bins=1000)

# Histogram of maximum denominator values
#plt.figure(7)
#plt.hist(d_max, bins=10000)

# Scatter Plot of maximum denominator values
#plt.figure(13)
#plt.scatter(n_max,d_max,c=epoch,marker='.',cmap='winter')

# Histogram of minimum denominator values
#plt.figure(9)
#plt.hist(np.log(d_min), bins=1000)

# Scatter Plot of minimum denominator values
#plt.figure(15)
#plt.scatter(n_max,np.log(d_min),c=epoch,marker='.',cmap='autumn')



""" RANGE 5 ANALYSIS """

# Load Data
data = pd.read_csv("range_file_5.csv")
data_extracted = pd.DataFrame(data, columns=['num_max','denom_max','denom_min','epoch']).to_numpy()

# Data arrays
n_max = data_extracted[:,0]
d_max = data_extracted[:,1]
d_min = data_extracted[:,2]
epoch = data_extracted[:,3]

# Formatting Color Data for epoch goes from Red -> Purple over time
normalized_epoch = epoch/np.max(epoch)
color_epoch = []
for e in normalized_epoch:
    color_epoch.append((1-e,0,1))

""" Untruncated Plots for Range 3 Data """
# Histogram of max numerator values
#plt.figure(6)
#plt.hist(n_max, bins=1000)

# Histogram of maximum denominator values
#plt.figure(7)
#plt.hist(d_max, bins=10000)

# Scatter Plot of maximum denominator values
#plt.figure(18)
#plt.scatter(n_max,d_max,c=epoch,marker='.',cmap='winter')

# Histogram of minimum denominator values
#plt.figure(9)
#plt.hist(np.log(d_min), bins=1000)

# Scatter Plot of minimum denominator values
#plt.figure(20)
#plt.scatter(n_max,np.log(d_min),c=epoch,marker='.',cmap='autumn')

""" Truncated Plots (outliers removed) """

""" SHOW ALL PLOTS """
plt.show()


