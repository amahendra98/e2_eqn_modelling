import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from D_set import D_set
import matplotlib.pyplot as plt


# Class net_from_array builds a sequaential network from D_in, list of #hidden nodes
# and type of activation function in each layer, D_out
def progress(total_epochs, epoch):
    if (epoch % (total_epochs/10) == 0):
        print("{}% Done".format(epoch/total_epochs * 100))

# Net architecture parameters
D_in = 2  # input dimension
D_out = 1  # output dimension
N = 2500  # batch size

# Specify hidden dimensions + activation functions
H = ('t',100,'t',100,'t',100,'t',100,'r')


# Initialize data
res = 50
r_min = 0
r_max = 2500
u_min = 1
u_max = 20000

trainset = D_set(res,r_min,r_max,u_min,u_max,rand=False)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=N, shuffle=True)

# Create models
model = torch.nn.Sequential(torch.nn.Linear(D_in, 100),
                            torch.nn.Tanh(),
                            torch.nn.Linear(100,100),
                            torch.nn.Tanh(),
                            torch.nn.Linear(100,100),
                            torch.nn.Tanh(),
                            torch.nn.Linear(100,100),
                            torch.nn.Tanh(),
                            torch.nn.Linear(100,100),
                            torch.nn.ReLU(),
                            torch.nn.Linear(100,D_out))
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
writer = SummaryWriter("runs/divider_neural_seq")
loss_fn = torch.nn.MSELoss()

#for name, param in model.named_parameters():
    #if param.requires_grad:
        #print(name, param.data)

#Training
print("TRAIN")
pointer = 0

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
for epoch in range(5000):
    model.train()
    for i, (x,y) in enumerate(trainloader):
        optimizer.zero_grad()

        y_pred = model(x).squeeze() #Squeeze solves MSELoss size mismatch error
        loss = loss_fn(y_pred,y)

        loss.backward()
        optimizer.step()

        err = loss.item()

        writer.add_scalar('training loss', err, epoch)
        progress(5000,epoch)
    #with torch.no_grad():
    #    for param in model.parameters():
    #        param -= learning_rate * param.grad