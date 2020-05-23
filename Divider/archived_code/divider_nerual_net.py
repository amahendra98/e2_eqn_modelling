import torch
import numpy as np
import common_functions as cf
from torch.utils.tensorboard import SummaryWriter

""" Code to generate a neural network implementation of division by testing different architectures """

# Class net_from_array builds a network from D_in, list of #hidden nodes
# and type of activation function in each layer, D_out
class net_from_array(torch.nn.Module):
    def __init__(self, D_in, D_out, H_list):
        super(net_from_array, self).__init__()
        self.linear_list = []
        n_inp = D_in
        for n in H_list:
            if (isinstance(n, str)):
                if n == 'r':
                    self.linear_list.append(torch.nn.ReLU())
                if n == 's':
                    self.linear_list.append(torch.nn.Sigmoid())
                if n == 't':
                    self.linear_list.append(torch.nn.Tanh())
            else:
                self.linear_list.append(torch.nn.Linear(n_inp, n))
                n_inp = n
        self.linear_list.append(torch.nn.Linear(n_inp, D_out))
        self.transforms = torch.nn.ModuleList(self.linear_list)

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x
def makeTagName(H):
    tag_name = ""
    for n in H:
        if isinstance(n, str):
            tag_name = tag_name + n
        else:
            tag_name = tag_name + str(n)
    return tag_name


# Net architecture and training parameters

D_in = 2  # input dimension
D_out = 1  # output dimension
N = 1000  # batch size
H = ('s', 100,'s',100,'s',100,'s',100,'s',100,'s',100,'r') # Specify hidden dimensions + activation functions
tag_name = makeTagName(H)  # Used for naming files and model runs

# Sampling x and y from domain
res = 100

r = np.random.uniform(0, 2500, res*res)
u = np.random.uniform(1, 20000, res*res)
x_data = torch.from_numpy(np.vstack((r,u)).transpose()).float()
y_data = torch.from_numpy(r/u).float()

"""                             Old Version of sampling Code
r = np.linspace(0,2500,res)
u = np.linspace(1,20000,res)
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


# Create models
model = net_from_array(D_in, D_out, H)
learning_rate = 1e-2
writer = SummaryWriter("runs/"+tag_name)
loss_fn = torch.nn.MSELoss()

#for name, param in model.named_parameters():
    #if param.requires_grad:
        #print(name, param.data)

#Training
print("TRAIN")
pointer = 0
for epoch in range(1000):

    # Collage batches
    x = x_data[pointer:pointer+N]
    y = y_data[pointer:pointer+N]
    pointer = (pointer+N)%(res*res)

    y_pred = model(x)
    y = y.unsqueeze(1)
    loss = loss_fn(y_pred,y)

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    # loss = torch.mean(torch.mean(((y_pred - y)**2)/(y*y), 1))
    err = loss.item()
    print(epoch,err)
    writer.add_scalar('training loss', err, epoch)
    cf.progress(5000,epoch)


# Evaluation, creates 3d graph of loss vs. input values over entire domain
print("\nEVAL")
cf.eval_domain_3d(u,r,model,clipAxis=True,loss_fn=1)





