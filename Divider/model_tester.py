import torch
import common_functions as cf
import div_models as div
from torch.utils.tensorboard import SummaryWriter

""" Code to generate a neural network implementation of division by testing different architectures """

# Net architecture and training parameters (not all variables used, depends on model)
D_in = 2  # input dimension
D_out = 1  # output dimension
N = 5000  # batch size
H = ('s', 100,'s',100,'s',100,'s',100,'s',100,'s',100,'r') # Specify hidden dimensions + activation functions
learning_rate = 1e-4
num_epochs = 5
trainable = True #only multiplier_reciprocator model is not trainable


# Create models and name the run
#model = div.net_from_array(D_in, D_out, H)     # Model for testing different neural nets
#tag_name = div.makeTagName(H)

#model = div.NALU(D_in, D_out)
#tag_name = "NALU"

#model = div.multiplier_reciprocator()
#tag_name = "multiplier_reciprocator"

model = div.adj_div_net()
tag_name = "adj_div_net"


# Sampling domain to produce input and output data
data_length = 10000   # Number of samples
u_min = 1     # u is the denominator
u_max= 20000
r_min = 0     # r is the numerator
r_max = 2500

(x_data, y_data) = cf.sampler(data_length,u_min,u_max,r_min,r_max)

# Save Training Data
writer = SummaryWriter("runs/"+tag_name)
loss_fn = torch.nn.MSELoss()


# Training
print("TRAIN")
pointer = 0
for epoch in range(num_epochs):
    # Collate batches
    x = x_data[pointer:pointer+N]
    y = y_data[pointer:pointer+N]
    pointer = (pointer+N)%(data_length)

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
    #print(epoch,err,y[0],y_pred[0])
    writer.add_scalar('training loss', err, epoch)
    cf.progress(num_epochs,epoch)


# Evaluation, creates 3d graph of loss vs. input values over entire domain
print("\nEVAL")
cf.eval_domain_3d(u_min,u_max,r_min,r_max,100,model,clipAxis=True,loss_fn=1)