# author:mingyue
# project:pythonProject
date: 2021 / 6 / 24
...
from    torch import nn, optim, autograd
import  torch
import pandas as pd
import numpy
h_dim = 400
batchsz = 588
class Generator(nn.Module):  #生成模型

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(7, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 7),
        )

    def forward(self, z):
        output = self.net(z)
        return output

net = Generator()
net.load_state_dict(torch.load('GAN_minmax.pkl'))
z = torch.randn(batchsz, 7)
xf = net(z)
print(xf.shape)
rawdata = pd.read_excel('./liqtrain_data.xlsx')  # 数据读入
t = rawdata.iloc[:, 0:]
t_max = t.max()
t_min = t.min()
t_max = torch.from_numpy(t_max.values)
t_min = torch.from_numpy(t_min.values)
print(t_max)
print(t_min)
xg = (xf*(t_max-t_min))+t_min
xg_real = xg.detach().numpy()
numpy.savetxt('GAN588.csv',xg_real,delimiter=',')
print(xg_real)
