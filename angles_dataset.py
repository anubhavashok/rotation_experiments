import torch
import numpy 
from torch.utils.data.dataset import Dataset  # For custom datasets
import math
import os
import numpy as np


def create_dataset(path):
    data = []
    for i in range(100000):
        axis = np.random.uniform(size=3)
        axis /= np.linalg.norm(axis)
        a = np.random.uniform(low=0, high=2*math.pi)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        data.append([x, y, z, a])
    np.save(path, data)

class AxisAngleDataset(Dataset):
    def __init__(self):
        path = './dataset.npy'
        if not os.path.exists(path):
            create_dataset(path)
        self.data = np.load(path)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]) 

    def __len__(self):
        return len(self.data)



def aa_to_mat(aa):
    axis, angle = aa[:3], aa[3]
    axis /= np.linalg.norm(axis)
    x = axis[0]
    y = axis[1]
    z = axis[2]
    c = math.cos(angle); s = math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])


def mat_to_quat(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q


def axisAngleGeodesicLossSingle(aa1, aa2):
    M = aa_to_mat(aa1)
    Mp = aa_to_mat(aa2)
    Mpp = np.dot(M, np.linalg.inv(Mp))
    inner = (Mpp[0, 0] + Mpp[1, 1] + Mpp[2, 2])/2.0
    inner = -1 if inner < -1 else inner
    inner = 1 if inner > 1 else inner
    l = math.acos(inner)
    return l


def axisAngleGeodesicLoss(aa1s, aa2s):
    loss = 0
    for i in range(aa1s.shape[0]):
        loss += axisAngleGeodesicLossSingle(aa1s[i], aa2s[i])
    return math.degrees(loss/aa1s.shape[0])

