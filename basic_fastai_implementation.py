import numpy as np
from torch import nn, optim
from torch.nn import Module, Linear, Embedding, functional as F
from fastai.column_data import ColumnarModelData
from fastai.model import fit
from torch.nn import functional as F

input_size, output_size, batch_size, epochs = 1, 1, 1, 20
x = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 3.1]
y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 1.3]
x = np.array(x).reshape((-1,1))
y = np.array(y).reshape((-1,1))
# Linear Regression Model
class model(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.l = nn.Linear(i, o)  
    
    def forward(self, x): return self.l(x)

m = model(input_size, output_size).cuda()
md = ColumnarModelData.from_arrays('.', [-1], x, y, bs=batch_size)
opt = optim.SGD(m.parameters(), 1e-4)
fit(m, md, epochs, opt, F.mse_loss)

input_size, output_size, batch_size, epochs = 1, 1, 1, 3

def lin(a,b,x): return a*x+b
def gen_fake_data(n, a, b):
    x = s = np.random.uniform(0,1,n) 
    y = lin(a,b,x) + 0.1 * np.random.normal(0,3,n)
    return x, y

x, y = gen_fake_data(10000, 3., 8.)

class model(nn.Module):
    def __init__(self, alist):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(alist[i], alist[i + 1]) for i in range(len(alist) - 1)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)
m = model([input_size, output_size]).cuda()
md = ColumnarModelData.from_arrays('.', [-1], x.reshape(-1,1), y.reshape(-1,1), bs=batch_size)
opt = optim.SGD(m.parameters(), 1e-1)
fit(m, md, epochs, opt, F.mse_loss)
