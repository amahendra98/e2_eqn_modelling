import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Sample domain to generate u, r, x_data, and y_data arrays
def sampler(res,umin,umax,rmin,rmax):
    r = np.random.uniform(rmin, rmax, res)
    u = np.random.uniform(umin, umax, res)

    x_data = torch.from_numpy(np.vstack((r, u)).transpose()).float()
    y_data = torch.from_numpy(r/u).float().unsqueeze(1)

    return (x_data,y_data)

# Prints percentage of epochs completed
def progress(total_epochs, epoch):
    if (epoch % (total_epochs/10) == 0):
        print("{}% Done".format(epoch/total_epochs * 100))

# Creates 3d Graph over domain
def eval_domain_3d(u_min,u_max,r_min,r_max,res,model,clipAxis=True, loss_fn=1):
    # u,r: arrays of values over the two axes, used in meshgrid calculation
    # clipAxis: Clips axes across domain u,r and z from 0 - 1
    # loss_fn: Determines what type of loss to plot -> 0 = MSELoss, 1 = Percent Error, 2 = Percent Error Squared
    u = np.linspace(u_min,u_max,res)
    r = np.linspace(r_min,r_max,res)
    X, Y = np.meshgrid(u, r)

    # Loss data storage
    loss_map = np.empty_like(X)
    avg_loss_y = np.empty_like(u)
    avg_loss_x = np.empty_like(avg_loss_y)

    for i in range(len(Y)): #Iterate through each Y value and calculate losses for all X permutations with that value
        x = X[i, :]
        y = Y[i, :]

        progress(len(Y), i)

        with torch.no_grad():
            y_pred = model(torch.tensor([x, y], dtype=torch.float32).t()).squeeze()
            y_actual = torch.tensor([x * y], dtype=torch.float32).squeeze()

            if loss_fn == 0:
                loss_map[i, :] = torch.pow(torch.sub(y_pred, y_actual), 2)  # MSELoss
            elif loss_fn == 1:
                loss_map[i, :] = torch.div(torch.abs(torch.sub(y_pred, y_actual)), y_actual)  # Percent Error
            elif loss_fn == 2:
                loss_map[i, :] = torch.pow(torch.div(torch.sub(y_pred, y_actual), y_actual), 2)  # Percent Error Squared

        avg_loss_y[i] = np.average(loss_map[i, :])

    print(loss_map)
    loss_map = loss_map.astype('float64')
    avg_loss = np.average(loss_map)
    for i in range(len(loss_map)):
        avg_loss_x[i] = np.average(loss_map[:, i])

    print("Average loss: " + str(avg_loss))
    print("Average loss r per u: " + ",".join(map(str, avg_loss_y)))
    print("Average loss u per r: " + ",".join(map(str, avg_loss_x)))

    # Plot graph
    plt.figure()
    ax = plt.axes(projection='3d')
    if clipAxis: #Use HardCoded limits
        ax.set(xlim=(u[0], u[-1]), ylim=(r[0], r[-1]), zlim=(0,1))
    ax.plot_surface(X, Y, loss_map, cmap='viridis', edgecolor='none')
    plt.show()