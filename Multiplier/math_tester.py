import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def derSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def secDerSigmoid(x):
    return derSigmoid(x)*(1-2*sigmoid(x))

u = 2
v = 3
r = u*v
lambdas = 0.00001
bias = 0.5
denom = 1/(4*lambdas*lambdas*secDerSigmoid(bias))

h = lambdas * np.array([u+v,-u-v,u-v,v-u]) + bias
h_s = sigmoid(h)
print(h_s)
out = np.dot(h_s, denom*np.array([1,1,-1,-1]))
print(out)
print(r)
