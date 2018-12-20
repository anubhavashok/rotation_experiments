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
    def __init__(self, mode):
        self.mode = mode
        path = './dataset.npy'
        if not os.path.exists(path):
            create_dataset(path)
        self.data = np.load(path)

    def __getitem__(self, idx):
        aa = self.data[idx]
        rep = aa
        if self.mode == "M":
            rep = aa_to_mat(aa).flatten().tolist()
        elif self.mode == "Q":
            rep = mat_to_quat(aa_to_mat(aa)).tolist()
        elif self.mode == "E":
            rep = mat_to_euler(aa_to_mat(aa)).tolist()
        return torch.FloatTensor(rep) 

    def __len__(self):
        return len(self.data)


def mat_to_euler(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def euler_to_mat(theta):
    R_x = np.array([[1,         0,                  0                   ],
        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
        ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
        [0,                     1,      0                   ],
        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
        ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
        [math.sin(theta[2]),    math.cos(theta[2]),     0],
        [0,                     0,                      1]
        ])

    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

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


def quat_to_mat(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
            [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def mat_to_aa(M):
    M = np.asarray(M, dtype=np.float)
    L, W = np.linalg.eig(M.T)
    i = np.where(np.abs(L - 1.0) < 1e-10)[0]
    direction = np.real(W[:, i[-1]]).squeeze()
    cosa = (np.trace(M) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (M[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (M[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (M[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return [direction[0], direction[1], direction[2], angle]


def quat_to_aa(q):
    return mat_to_aa(quat_to_mat(q))


def euler_to_aa(e):
    return mat_to_aa(euler_to_mat(e))

def to_aa(d, mode):
    if mode == "Q":
        return quat_to_aa(d)
    elif mode == "M":
        return mat_to_aa(d)
    elif mode == "E":
        return euler_to_aa(d)
    return d



def axisAngleGeodesicLossSingle(aa1, aa2):
    M = aa_to_mat(aa1)
    Mp = aa_to_mat(aa2)
    Mpp = np.dot(M, np.linalg.inv(Mp))
    inner = (Mpp[0, 0] + Mpp[1, 1] + Mpp[2, 2])/2.0
    inner = -1 if inner < -1 else inner
    inner = 1 if inner > 1 else inner
    l = math.acos(inner)
    return l


def geodesicLoss(aa1s, aa2s, mode):
    loss = 0
    for i in range(aa1s.shape[0]):
        aa1 = to_aa(aa1s[i], mode)
        aa2 = to_aa(aa2s[i], mode)
        loss += axisAngleGeodesicLossSingle(aa1, aa2)
    return math.degrees(loss/aa1s.shape[0])

