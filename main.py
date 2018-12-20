import torch
from angles_dataset import *
from torch import nn, optim
from logger import Logger
import time
from torch.autograd import Variable

mode = "E"
batch_size = 64

N = {"Q":4, "AA":4, "E": 3, "M":9}[mode]
net = nn.Sequential(
        nn.Linear(N, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 128),
        nn.LeakyReLU(),
        nn.Linear(128, N)
        )


dataset = torch.utils.data.DataLoader(AxisAngleDataset(mode), batch_size=batch_size, shuffle=False)
logger = Logger('./logs/'+str(time.time())+'/')
l2Loss = nn.MSELoss()

def train():
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    for step, (data) in enumerate(dataset):
        #if step == int(10000.0/batch_size):
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = 1e-6
        pred = net(Variable(data))
        loss = l2Loss(pred, Variable(data).detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        d = data.cpu().numpy()
        p = pred.cpu().detach().numpy()
        geoLoss = geodesicLoss(d, p, mode)
        info = {'l2Loss': loss.item(), 'geodesicLoss': geoLoss}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)


train()
