import numpy as np
import math
import matplotlib.pyplot as plt

#Same Code from math tester except using x^4
def secDer(x):
    return 12*x*x
def func(x):
    return np.power(x,4)

u = np.array(range(1,600))
u_1 = np.arange(0.01,1,0.01)
u = np.concatenate((u_1,u))
v = 1/u

lambdas = 0.0001
bias = 1
denom = 4*lambdas*lambdas*secDer(bias)

h = lambdas * np.array([u+v,-u-v,u-v,v-u]) + bias
h_s = func(h)
f1 = np.array([1]*698)
f2 = np.array([-1]*698)
z = np.vstack((f1,f1,f2,f2))
out = np.dot(h_s.transpose(), z/denom)
#plt.plot(u,out)
#plt.show()
np.savetxt('out.csv', out, delimiter=',')
np.savetxt('u.csv', u, delimiter=',')

#Division Implementation
u = 10
r = 1

lambdas = 0.0001
bias = 1
denom = 4*lambdas*lambdas*secDer(bias)
#denom_s = (3.3333340721419910e-005*u*u + -5.3400962818361819e-009*u + 1.0000008073636928)*secDer(bias) #0.01
denom_s = (3.3333351668477530e-009*u*u + -1.2988072963521636e-012*u + 1.0000000002043055)*secDer(bias) #0.0001
#denom_s = (9.0219060873416211e-020*u*u*u*u -1.0719126394472148e-016*u*u*u + 3.3333759841999366e-009*u*u - 6.6489097032395681e-012*u + 1.0000000003613836)*secDer(bias) #0.0001

u2_v2 = denom_s/(4*lambdas*lambdas) - 3*bias*bias/(lambdas*lambdas)                       #Gives u^2 + v^2
u_v2 = u2_v2 + 2*r                                                          #Gives (u+v)^2 = u^2 + v^2 +2uv
v = math.sqrt(u_v2) - u
print(v)

