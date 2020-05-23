import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

""" All Divider models in test """

# Model: net_from_array

# builds a normal neural network from D_in, list of #hidden nodes and
# type of activation function in each layer, D_out
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

# Specifically used with net_from_array
def makeTagName(H):
    tag_name = ""
    for n in H:
        if isinstance(n, str):
            tag_name = tag_name + n
        else:
            tag_name = tag_name + str(n)
    return tag_name




# Model: NALU

# Architecture specifically designed for multiplication and division composed of NAC model
# model taken from https://towardsdatascience.com/a-quick-introduction-to-neural-arithmetic-logic-units-288da7e259d7
class NAC(torch.nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.W_hat = Parameter(torch.Tensor(n_out, n_in))
    self.M_hat = Parameter(torch.Tensor(n_out, n_in))
    #self.reset_parameters()

  #def reset_parameters(self):
    #init.kaiming_uniform_(self.W_hat)
    #init.kaiming_uniform_(self.M_hat)

  def forward(self, input):
    weights = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
    return F.linear(input, weights)

class NALU(torch.nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.NAC = NAC(n_in, n_out)
    self.G = Parameter(torch.Tensor(1, n_in))
    self.eps = 1e-6

  def forward(self, input):
    g = torch.sigmoid(F.linear(input, self.G))
    y1 = g * self.NAC(input)
    y2 = (1 - g) * torch.exp(self.NAC(torch.log(torch.abs(input) + self.eps)))
    return y1 + y2




# Model: multiplier_reciprocator

# Untrainable pseudo-neural network designed to do division by combining
# hard coded multiplication and reciprocation

def secDerSigmoid(x):
    return torch.sigmoid(x)*(-1*torch.sigmoid(x) + 1)*(-2*torch.sigmoid(x) + 1)

# Multiplier based off of paper:
# Lin, H. W., Tegmark, M., & Rolnick, D. (2017). Why Does Deep and Cheap Learning Work So Well?
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

# Reciprocator reverse engineered from multiplicator
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

# Divider neural network combining both
class multiplier_reciprocator(torch.nn.Module):
    def __init__(self):
        super(multiplier_reciprocator, self).__init__()
        self.recip = reciprocator()
        self.mult = multiplier()

    def forward(self,x):
        r = x[:,0].double()
        u = x[:,1].double()
        inv = self.recip(u).squeeze(1)
        fix_dim = inv.size()[0]
        inv = inv.view(fix_dim,1)
        r = r.view(fix_dim,1)
        inv = torch.cat((inv,r),1)
        v = self.mult(inv.t()).view(fix_dim,1).float()
        return v




# Model: adj_div_net

# Trainable model, constructed out of 3 neural nets each with desginated purposes
# Model described in paper: Pei, Jin-Song & Mai, Eric & Wright, Joseph. (2008).
# "Mapping Some Functions and Four Arithmetic Operations to Multilayer Feedforward Neural Networks"
# https://www.researchgate.net/publication/278329389_Mapping_Some_Functions_and_Four_Arithmetic_Operations_to_Multilayer_Feedforward_Neural_Networks

class net1_approximator(torch.nn.Module):
    def __init__(self):
        super(net1_approximator,self).__init__()
        self.lin1 = torch.nn.Linear(1,3)
        self.lin2 = torch.nn.Linear(3,1)
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        return self.lin2(self.sig(self.lin1(x)))

class net2_reciprocator(torch.nn.Module):
    def __init__(self):
        super(net2_reciprocator,self).__init__()
        self.lin1 = torch.nn.Linear(1,2)
        self.lin2 = torch.nn.Linear(2,1)
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        return self.lin2(self.sig(self.lin1(x)))

class net3_multiplicator(torch.nn.Module):
    def __init__(self):
        super(net3_multiplicator,self).__init__()
        self.lin1 = torch.nn.Linear(2,8)
        self.lin2 = torch.nn.Linear(8,1)
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        return self.lin2(self.sig(self.lin1(x)))

class adj_div_net(torch.nn.Module):
    def __init__(self):
        super(adj_div_net,self).__init__()
        self.dict = torch.nn.ModuleDict([
            ['net1', net1_approximator()],
            ['net2', net2_reciprocator()],
            ['net3', net3_multiplicator()]
        ])
        self.relu = torch.nn.ReLU()

    def forward(self, input):

        r = input[:,0].unsqueeze(1)
        u = input[:,1].unsqueeze(1)

        approx = self.relu((self.dict['net1'](r)))
        recip = self.relu((torch.mul(self.dict['net2'](u),100)))

        setupInp = torch.cat((approx,recip),1)

        ans = self.dict['net3'](setupInp)
        return ans