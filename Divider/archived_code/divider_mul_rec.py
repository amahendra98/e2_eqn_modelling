import torch
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter

def secDerSigmoid(x):
    return torch.sigmoid(x)*(-1*torch.sigmoid(x) + 1)*(-2*torch.sigmoid(x) + 1)

class reciprocator(torch.nn.Module):
    def __init__(self):
        super(reciprocator,self).__init__()
        self.lambdas = 0.0001
        self.b = 1

        self.c1 = torch.tensor([[1.0000000002043055]], dtype=torch.float64)
        self.c2 = torch.tensor([[-1.2988072963521636e-012]], dtype=torch.float64)
        self.c3 = torch.tensor([[3.3333351668477530e-009]], dtype=torch.float64)

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

class multiplier(torch.nn.Module):
    def __init__(self):
        super(multiplier,self).__init__()
        self.lambdas = Parameter(torch.tensor([[0.0001]]))
        self.bias_1 = Parameter(torch.tensor([[1]], dtype=torch.float64))
        self.o2 = Parameter(12*self.bias_1*self.bias_1)

        arr_1 = [[1,1],[-1,-1],[1,-1],[-1,1]]
        arr_2 = [[1,1,-1,-1]]

        self.weight_1 = Parameter(torch.tensor(arr_1, dtype=torch.float64))
        self.sig = torch.nn.Sigmoid()
        self.weight_2 = Parameter(torch.tensor(arr_2, dtype=torch.float64))

    def forward(self, input):
        h1 = self.sig(self.weight_1.mm(input)*self.lambdas + self.bias_1)
        denom = 4*self.lambdas*self.lambdas*self.o2
        return self.weight_2.mm(h1)/denom

class divider(torch.nn.Module):
    def __init__(self):
        super(divider, self).__init__()
        self.recip = reciprocator()
        self.mult = multiplier()

    def forward(self,u,r):
        inv = self.recip(u).squeeze(1)
        inv = torch.cat((inv,r),0).unsqueeze(1)
        inv[1] = r.data
        v = self.mult(inv)
        return v

# Specify Net
D_in = 2  # input dimension
D_out = 1  # output dimension
model = divider()
loss_fn = torch.nn.MSELoss(reduction='mean')
writer = SummaryWriter("../runs/divider_net")

for epoch in range(50):
    u = torch.mul(torch.rand([1,1], dtype=torch.float64),20000)
    u = u[0]
    r = torch.mul(torch.rand([1,1], dtype=torch.float64),2500)
    r = r[0]

    y_pred = model(u,r)
    y = torch.div(r,u)
    y = y.view([1,1])
    loss = loss_fn(y_pred,y)
    err = loss.item()

    writer.add_scalar('training loss', err, epoch)
    print(err, y_pred.item(), y.item())

    model.zero_grad()
    loss.backward()



