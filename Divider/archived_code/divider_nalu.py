import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init
import numpy as np

""" Implementation of NALU, model taken from 
https://towardsdatascience.com/a-quick-introduction-to-neural-arithmetic-logic-units-288da7e259d7 """

class NAC(torch.nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.W_hat = Parameter(torch.Tensor(n_out, n_in))
    self.M_hat = Parameter(torch.Tensor(n_out, n_in))
    #self.reset_parameters()

  #def reset_parameters(self):
    #init.kaiming_uniform_(self.W_hat)
    #init.kaiming_uniform_(self.M_hat)

  def forward(self, input):
    weights = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
    return F.linear(input, weights)

class NALU(torch.nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.NAC = NAC(n_in, n_out)
    self.G = Parameter(torch.Tensor(1, n_in))
    self.eps = 1e-6

  def forward(self, input):
    g = torch.sigmoid(F.linear(input, self.G))
    y1 = g * self.NAC(input)
    y2 = (1 - g) * torch.exp(self.NAC(torch.log(torch.abs(input) + self.eps)))
    return y1 + y2

D_in = 2  # input dimension
D_out = 1  # output dimension
N = 100

# Initialize Data
res = 100
#r = np.linspace(0,2500,res)
#u = np.linspace(1,20000,res)


r = np.random.uniform(0, 2500, res*res)
u = np.random.uniform(1, 20000, res*res)
seed = np.random.randint()
np.random.seed(seed)
np.random.shuffle(r)
np.random.seed(seed)
np.random.shuffle(u)

x_data = torch.from_numpy(np.vstack((r,u)).transpose()).float()
y_data = torch.from_numpy(r/u).float()

"""
x_np = np.empty((res*res,2))
y_np = np.empty((res*res,1))

for r_index in range(res):
    x_np[res*r_index : res*r_index + res,0] = np.ones_like(r)*r[r_index]
    x_np[res*r_index : res*r_index + res,1] = u

np.random.shuffle(x_np)

y_np = x_np[:,0]/x_np[:,1]

x_data = torch.from_numpy(x_np).float()
y_data = torch.from_numpy(y_np).float()
"""

# Setup model
model = NALU(D_in, D_out)
loss_fn = torch.nn.MSELoss(reduction='mean')
learning_rate = 1e-1
writer = SummaryWriter("../runs/NALU")

# Training
pointer = 0
for epoch in range(2000):
    #x = x_data
    #y = y_data
    x = x_data[pointer:pointer+N]
    y = y_data[pointer:pointer+N]
    pointer = (pointer + N)%(res*res)

    y_pred = model(x).squeeze()
    #loss = torch.mean(torch.mean(torch.pow((y_pred - y),2), 1))
    loss = loss_fn(y_pred,y)
    err = loss.item()

    writer.add_scalar('training loss', err, epoch)
    # writer.add_graph(model,x) Get this working for net visualization
    # y_pred has extra dim to be squeezed
    print(epoch, err)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            #print(param.grad)
            param -= learning_rate * param.grad