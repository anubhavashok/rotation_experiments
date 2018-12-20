import torch
from angles_dataset import *
from torch import nn, optim
from logger import Logger

N = 4
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


dataset = torch.utils.data.DataLoader(AxisAngleDataset(), batch_size=64, shuffle=False)
logger = Logger('./logs/'+str(time.time())+'/')
l2Loss = nn.MSELoss()

def train():
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    for step, (data) in enumerate(dataset):
        pred = net(Variable(data))
        loss = l2Loss(pred, Variable(data).detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        geoLoss = axisAngleGeodesicLoss(data.cpu().numpy(), pred.cpu().detach().numpy())
        info = {'l2Loss': loss, 'geodesicLoss': geoLoss}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)


train()
