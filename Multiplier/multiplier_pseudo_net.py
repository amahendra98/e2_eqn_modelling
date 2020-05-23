import torch
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter

def secDerSigmoid(x):
    return torch.sigmoid(x)*(-1*torch.sigmoid(x) + 1)*(-2*torch.sigmoid(x) + 1)

class multiplier(torch.nn.Module):
    def __init__(self):
        super(multiplier,self).__init__()
        self.lambdas = Parameter(torch.tensor([[0.00001]]))
        self.bias_1 = Parameter(torch.tensor([[0.5]], dtype=torch.float64))
        self.o2 = Parameter(secDerSigmoid(self.bias_1))

        arr_1 = [[1,1],[-1,-1],[1,-1],[-1,1]]
        arr_2 = [[1,1,-1,-1]]

        self.weight_1 = Parameter(torch.tensor(arr_1, dtype=torch.float64))
        self.sig = torch.nn.Sigmoid()
        self.weight_2 = Parameter(torch.tensor(arr_2, dtype=torch.float64))

    def forward(self, input):
        h1 = self.sig(self.weight_1.mm(input)*self.lambdas + self.bias_1)
        denom = 4*self.lambdas*self.lambdas*self.o2
        return self.weight_2.mm(h1)/denom

# Dimensions
D_in = 2                                         #input dimension
D_out = 1                                      #output dimension
H = 4                                           #hidden dimension

# Specify Net
model = multiplier()
loss_fn = torch.nn.MSELoss()
writer = SummaryWriter("runs/first")

for epoch in range(50):
    x = torch.randn([2,1], dtype=torch.float64)
    u = x[0]
    v = x[1]
    r = u*v


    y_pred = model(x)
    y = torch.tensor(r, dtype=torch.float64)
    y = y.view([1,1])
    loss = loss_fn(y_pred,y)
    err = loss.item()

    writer.add_scalar('training loss', err, epoch)
    print(epoch, err, y_pred, y)

    model.zero_grad()
    loss.backward()

    #with torch.no_grad():
    #    count = 0
    #    for param in model.parameters():
    #        if count < 1:
    #            param -= learning_rate * param.grad
    #            print(param)
    #        count += 1


