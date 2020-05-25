import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
from torch.nn.functional import linear

#Dimensions and Hyperparameters
D_in = 2
H1 = 5
H2 = 8
D_out = 1
learning_rate = 1e-3
N = 10000

class paper_divider(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(paper_divider,self).__init__()
        self.linear1 = Parameter(torch.rand((H1,D_in)))
        with torch.no_grad():
            self.linear1[0][1] = 0
            self.linear1[1][1] = 0
            self.linear1[2][1] = 0
            self.linear1[3][0] = 0
            self.linear1[4][0] = 0
        self.bias = Parameter(torch.rand((1,1)))
        self.lin2 = torch.nn.Linear(H1, H2)
        self.lin3 = torch.nn.Linear(H2,D_out)
        self.sig = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        return self.relu(self.lin3(self.sig(self.lin2(self.sig(linear(input,self.linear1,self.bias))))))

# Specify Net
model = paper_divider(D_in,H1,H2,D_out)
writer = SummaryWriter("../runs/paper_divider_net")

# Initialize Data
res = 100
#r = np.linspace(0,2500,res)
#u = np.linspace(1,20000,res)

r = np.random.uniform(0, 2500, res*res)
u = np.random.uniform(1, 20000, res*res)

x_data = torch.from_numpy(np.vstack((r,u)).transpose()).float()
y_data = torch.from_numpy(r/u).float()

""""
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

# Train
loss_fn = torch.nn.MSELoss()
pointer = 0
for epoch in range(1000):
    x = x_data[pointer:pointer+N]
    y = y_data[pointer:pointer+N]
    pointer = (pointer+N)%10000

    y_pred = model(x)
    #loss = torch.mean(torch.mean(((y_pred - y)**2)/(y*y), 1))
    loss = loss_fn(y_pred.squeeze(),y)
    err = loss.item()

    writer.add_scalar('training loss', err, epoch)
    # writer.add_graph(model,x) Get this working for net visualization
    if epoch % 20 == 19:
        print(epoch, err)

    model.zero_grad()
    loss.backward()
    count = 0
    with torch.no_grad():
        for param in model.parameters():
            #print(count, param, param.grad)
            if count == 0:
                param -= learning_rate * param.grad
                param[0][1] = 0
                param[1][1] = 0
                param[2][1] = 0
                param[3][0] = 0
                param[4][0] = 0
                count = count + 1
            else:
                param -= learning_rate * param.grad
                count = count + 1
