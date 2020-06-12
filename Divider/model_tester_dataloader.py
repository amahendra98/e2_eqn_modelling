import torch
import numpy as np
import common_functions as cf
import div_models as div
from torch.utils.tensorboard import SummaryWriter
from D_set import Divider_Domain_set as D
from torch.utils.data import DataLoader

""" Code to generate a neural network implementation of division by testing different architectures """

# Net architecture and training parameters (not all variables used, depends on model)
D_in = 2  # input dimension
D_out = 1  # output dimension
N = 5000  # batch size
H = ('s', 100,'s',100,'s',100,'s',100,'s',100,'s',100,'r') # Specify hidden dimensions + activation functions
learning_rate = 1e-4
num_epochs = 1000
trainable = True #only multiplier_reciprocator model is not trainable


# Create models and name the run
model = div.net_from_array(D_in, D_out, H)     # Model for testing different neural nets
tag_name = div.makeTagName(H)

#model = div.NALU(D_in, D_out)
#tag_name = "NALU"

#model = div.multiplier_reciprocator()
#tag_name = "multiplier_reciprocator"

#model = div.adj_div_net()
#tag_name = "adj_div_net"


# Sampling domain to produce input and output data
data_length = 10000   # Number of samples
u_min = 1     # u is the denominator
u_max= 20000
r_min = 0     # r is the numerator
r_max = 2500
rand = True

D = DataLoader(D(100, r_min, r_max, u_min, u_max, rand), batch_size = N,shuffle=True)


# Save Training Data
writer = SummaryWriter("runs/"+tag_name)
loss_fn = torch.nn.MSELoss()


# Training
print("TRAIN")
num_batches = data_length/N
train_loss = []
for e in range(int(num_epochs/num_batches)):
    for i, sampled_batch in enumerate(D):           # For loop runs data_length/N times
        epoch = e*num_batches + i
        x = sampled_batch['x_data'].squeeze().float()
        y = sampled_batch['y_data'].float()

        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        # loss = torch.mean(torch.mean(((y_pred - y)**2)/(y*y), 1))

        if trainable:
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

        err = loss.item()
        print(epoch,err)
        writer.add_scalar('training loss', err, epoch)
        cf.progress(num_epochs/num_batches,e)


# Evaluation, creates 3d graph of loss vs. input values over entire domain
print("\nEVAL")
cf.eval_domain_3d(u_min,u_max,r_min,r_max,100,model,clipAxis=True,loss_fn=1)