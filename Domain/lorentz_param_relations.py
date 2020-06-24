import numpy as np
import matplotlib.pyplot as plt

def e2_num_denom(w0,wp,g,w):
    num = wp*wp*g*w
    denom = np.power(w0*w0 - np.power(w,2), 2) + g*g*np.power(w,2)
    q = num/denom
    return [num,denom,q]

def fwp(r):
    return r*np.sqrt(np.pi)/100
def fg(r):
    return r*np.sqrt(np.pi)/1000
def fw0(h):
    return 100 / h

r = np.linspace(20,200,1000)
h = np.linspace(22,100,1000)
w = np.linspace(0.5,5,300)

wp = r*np.sqrt(np.pi)/100
g = r*np.sqrt(np.pi)/1000
w0 = 100 / h

val_array = np.zeros((1000*1000*300,3))

index = 0
for wval in w:
    print(index)
    for rval in r:
        for hval in h:
            val_array[index] = np.array( e2_num_denom( fw0(hval), fwp(rval), fg(rval), wval) ).transpose()
            index = index + 1

fig1 = plt.figure()
plt.title("Numerator-denominator pairs Normal Space")
plt.scatter(val_array[:,0],val_array[:,1])
plt.xlabel("num")
plt.ylabel("denom")

fig2 = plt.figure()
plt.title("Numerator-denominator pairs Log Space")
plt.scatter(np.log10(val_array[:,0]),np.log10(val_array[:,1]))
plt.xlabel("log10(num)")
plt.ylabel("log10(denom)")
plt.show()

fig3 = plt.figure()
plt.title("Quotient Histogram")
plt.hist(val_array[:,2],bins=10000)
plt.show()

fig4 = plt.figure()
plt.title("Log Quotient Histogram")
plt.hist(np.log10(val_array[:,2]),bins=10000)
plt.show()