import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def merge(distn, sorted_data):
     new_data = np.zeros((len(distn),5))
     idx = 0
     for d_idx in range(len(distn)):
        lock_val = distn[d_idx]
        q = lock_val
        while lock_val > sorted_data[idx,2]:
            idx = idx + 1
        q = sorted_data[idx,2]
        z = idx
        new_data[d_idx] = [sorted_data[idx,0], sorted_data[idx,1], q, (q-lock_val)/q, z]

     return new_data

def stage_one_sampler(num_min, num_max, denom_min, denom_max, length, schema):

    denom_min = denom_min - denom_min/2 # Pads upper bound with restricting num, denom domain space

    if schema == 'u_log':
        num_min = np.log10(num_min)
        num_max = np.log10(num_max)
        denom_min = np.log10(denom_min)
        denom_max = np.log10(denom_max)

    num = np.random.uniform(num_min, num_max, length)
    denom = np.random.uniform(denom_min, denom_max, length)

    if schema == 'u_log':
        quotient = num - denom
        quotient = 10 ** quotient
        num = 10 ** num
        denom = 10 ** denom
    if schema == 'unf':
        quotient = num/denom

    data = np.c_[num, denom, quotient]
    return data

def stage_two_sampler(data, qmin, qmax, length, schema=0):
    # Pre-sort code to eliminate undesired denom and num pairings goes here

    sorted = data[np.argsort(data[:,2])]
    #distn = stats.uniform.rvs(size=length,loc=qmin,scale=qmax-qmin)
    distn = stats.loguniform.rvs(a=qmin,b=qmax,size=length)
    distn = np.sort(distn)
    sample = merge(distn,sorted)
    return sample

def sampler(nmin,nmax,dmin,dmax,s1,l1,s2,l2):
    data = stage_one_sampler(nmin,nmax,dmin,dmax,l1,s1)
    qmin = nmin/dmax
    qmax = nmax/dmin
    return stage_two_sampler(data,qmin,qmax,l2,s2)
'''
sample = sampler(0.001,10,0.0001,1000,'u_log',10000000,'unf',100000)

#print("Mean Percent Error:", np.average(sample[:,3]))

fig1 = plt.figure()
plt.title("Number of points in sample 1 less than value in sample 2")
ax = plt.axes()
ax.plot(sample[:,2],sample[:,4])
plt.xlabel("Value at sample 2")
plt.ylabel("# of values in sample 1")

fig2 = plt.figure()
plt.hist(sample[:,2], bins=100000)
plt.title("Histogram of values over distribution")
plt.xlabel("Value in sample bin size = 100000")
plt.ylabel("# Occurences")

fig3 = plt.figure()
plt.hist(np.log10(sample[:,2]), bins=100000)
plt.title("Histogram of log scaled values over distribution")
plt.xlabel("Value in sample bin size = 100000")
plt.ylabel("# Occurences")

plt.show()
'''

sample = stage_one_sampler(0.001,10,0.0001,1000,10000000,'unf')

fig2 = plt.figure()
plt.hist(sample[:,2], bins=100000)
plt.title("Histogram of values over distribution")
plt.xlabel("Value in sample bin size = 100000")
plt.ylabel("# Occurences")

fig3 = plt.figure()
plt.hist(np.log10(sample[:,2]), bins=100000)
plt.title("Histogram of log scaled values over distribution")
plt.xlabel("Value in sample bin size = 100000")
plt.ylabel("# Occurences")
plt.show()

#sampler(1,10,1,10,'u_log',1000,'unf',10)