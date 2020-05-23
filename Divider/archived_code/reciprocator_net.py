import torch
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter

class reciprocator(torch.nn.Module):
    def __init__(self):
        super(reciprocator,self).__init__()
        self.lambdas = 0.0001
        self.b = 1

        self.c1 = Parameter(torch.tensor([[1.0000000002043055]], dtype=torch.float64))
        self.c2 = Parameter(torch.tensor([[-1.2988072963521636e-012]], dtype=torch.float64))
        self.c3 = Parameter(torch.tensor([[3.3333351668477530e-009]], dtype=torch.float64))

        self.w_denom = Parameter(torch.tensor([[1/(4*self.lambdas*self.lambdas)]], dtype=torch.float64))
        self.bias_denom = Parameter(torch.tensor([[-3*self.b*self.b/(self.lambdas*self.lambdas)]], dtype=torch.float64))

        self.bias_u2_v2 = Parameter(torch.tensor([[2]], dtype=torch.float64))

    def forward(self, u):
        # (c3 * u * u + c2 * u + c1)*12*b*b
        denom_s = torch.mul(torch.add(torch.sub(torch.mul(torch.mul(u,u),self.c3),torch.mul(u,self.c2)),self.c1), 12*self.b*self.b)
        u2_v2= torch.add(self.w_denom.mm(denom_s), self.bias_denom)
        u_v2 = u2_v2 + self.bias_u2_v2
        u_v = torch.sqrt(u_v2)
        v = torch.sub(u_v, u)
        return v

# Specify Net
model = reciprocator()
loss_fn = torch.nn.MSELoss()
writer = SummaryWriter("runs/first")

for epoch in range(50):
    x = torch.mul(torch.rand([1,1], dtype=torch.float64),600)
    u = x[0]
    r = 1/u

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
    #        if count < 3:
    #            param -= learning_rate * param.grad
    #            print(param)
    #        count += 1


