import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Lorentz Imaginary Component Function
def e2(w0,wp,g,w):
    return wp*wp*g*w/(np.power(w0*w0 - np.power(w,2), 2) + g*g*np.power(w,2))

# Load Data
w = np.linspace(0.8,1.5,num=300)

# Dimensions
D_in = 3                                         #input dimension
D_out = 300                                      #output dimension
H = 5000                                           #hidden dimension

# Specify Net
model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.Sigmoid(), torch.nn.Linear(H, D_out))
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
writer = SummaryWriter("runs/Hidden_{}".format(H))

for epoch in range(500):
    w0 = 5 * np.random.random()
    wp = 5 * np.random.random()
    g = np.random.random()

    x = torch.tensor([w0,wp,g])

    y_pred = model(x)
    y = torch.tensor(e2(w0,wp,g,w), dtype=torch.float32)
    loss = loss_fn(y_pred,y)
    err = loss.item()

    writer.add_scalar('training loss', err, epoch)
    if epoch % 20 == 19:
        print(epoch, err)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad


