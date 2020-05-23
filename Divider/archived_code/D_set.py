import torch
import numpy as np

class D_set(torch.utils.data.Dataset):
    def __init__(self, res, r_min, r_max, u_min, u_max, rand):
        self.res = res
        if rand:
            self.r = np.random.uniform(r_min, r_max, size=(res,0))
            self.u = np.random.uniform(u_min, u_max, size=(res,0))
        else:
            self.r = np.linspace(r_min, r_max, res)
            self.u = np.linspace(u_min, u_max, res)

    def __len__(self):
        return self.res*self.res

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = np.array(idx.tolist())

        r_row = self.r[np.mod(idx, self.res)]
        u_row = self.u[np.trunc(idx / self.res).astype(int)]
        v = r_row/u_row

        inp = np.vstack((r_row, u_row)).transpose()
        inp = torch.from_numpy(inp).float()

        return inp, v

