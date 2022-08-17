import  torch
from    torch import nn, optim, autograd
import  numpy as np
import  visdom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from    torch.nn import functional as F
from    matplotlib import pyplot as plt
import random

h_dim = 400
batchsz = 64
viz = visdom.Visdom()   

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


class Discriminator(nn.Module):  #判别模型

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(7, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

def data_generator():
    rawdata = pd.read_excel('./data2.xlsx')  # 数据读入
    t = rawdata.iloc[:, 0:]
    liquefaction =(t-t.min())/(t.max()-t.min())
    print(liquefaction)
    liquefactionNP = liquefaction.values
    while True:
        dataset = []
        a = random.sample(range(0, 195), 64)
        for i in range(64):
            dataset.append(liquefactionNP[a[i]])
        dataset = np.array(dataset, dtype='float32')
        #print(dataset)
        yield dataset


def weights_init(m):  #权重初始化
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def gradient_penalty(D, xr, xf):    #惩罚
    """

    :param D:
    :param xr:
    :param xf:
    :return:
    """
    LAMBDA = 0.2

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, 7]
    alpha = torch.rand(batchsz, 1)
    alpha = alpha.expand_as(xr)

    interpolates = alpha * xr + ((1 - alpha) * xf)
    #设置需要导数信息
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp

def main():

    torch.manual_seed(23)
    np.random.seed(23)

    G = Generator()
    D = Discriminator()
    G.apply(weights_init)
    D.apply(weights_init)

    optim_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    data_iter = data_generator()
    print('batch:', next(iter(data_iter)).shape)

    viz.line([[0,0]], [0], win='loss', opts=dict(title='loss',
                                                 legend=['D', 'G']))

    for epoch in range(300000):

        # 1. train discriminator for k steps
        for _ in range(5):
            #real data
            xr = next(data_iter)
            xr = torch.from_numpy(xr)  #转换成tensor


            # [b]
            predr = (D(xr))
            # max log(lossr)
            lossr = -predr.mean()

            # [b, 7]
            z = torch.randn(batchsz, 7)
            # stop gradient on G
            # [b, 7]
            xf = G(z).detach()
            # [b]
            predf = (D(xf))
            # min predf
            lossf = predf.mean()

            # gradient penalty
            gp = gradient_penalty(D, xr, xf.detach())

            loss_D = lossr + lossf + gp
            optim_D.zero_grad()
            loss_D.backward()
            # for p in D.parameters():
            #     print(p.grad.norm())
            optim_D.step()


        # 2. train Generator
        z = torch.randn(batchsz, 7)
        xf = G(z)
        predf = (D(xf))
        # max predf
        loss_G = -predf.mean()
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()



        if epoch % 3000 == 0:

            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')

            print(loss_D.item(), loss_G.item())

    torch.save(G.state_dict(), 'GAN_minmax.pkl')






if __name__ == '__main__':
    main()