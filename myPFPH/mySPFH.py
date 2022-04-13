import torch

import pfh.utils as utils
from pfh.pfh import PFH, SPFH, FPFH
from time import time


# source_pc = np.random.rand(1024, 3)
source_pc = utils.load_pc_np('data/plant_source.npy')
# print(source_pc)

# Run ICP with some example parameters
et, div, nneighbors, rad = 0.1, 2, 8, 0.03
Icp = SPFH(et, div, nneighbors, rad)   # Fast PFH
normS, indS = Icp.calc_normals(source_pc)
spfhHist = Icp.calcHistArray(source_pc, normS, indS)

# Icp = SPFH(et, div, nneighbors, rad)   # Fast PFH
# normS, indS = Icp.calc_normals(source_pc)
# fpfhHist = Icp.calcHistArray(source_pc)

def transformFormat(originPC):
    pointcloud = torch.zeros((1, len(originPC), 3))
    for idx, val in enumerate(originPC):
        pointcloud[0, idx] = torch.from_numpy(val).flatten()
    return pointcloud


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

batch_pc_torch = transformFormat(source_pc)
neighborIdx = knn(batch_pc_torch.transpose(2, 1), 9)[0, :, 1:]

pc_torch = batch_pc_torch[0]
Amat = torch.zeros(pc_torch.shape[0],pc_torch.shape[0])



neighborPoint = pc_torch[neighborIdx.reshape(-1), :].reshape(-1, 8, 3)
# 计算邻居点



normal_torch = torch.zeros(pc_torch.shape)
for i, X in enumerate(neighborPoint):
    X = X - X.mean(axis=0)
    cov = torch.matmul(X.T, X)/8
    _, _, Vt = torch.linalg.svd(cov)
    normal = Vt[2, :]
    if torch.matmul(normal, -1.*(pc_torch[i])) < 0:
        normal = -1.*normal
    normal_torch[i] = normal
# 计算法向量

neighborNormal = normal_torch[neighborIdx.reshape(-1), :].reshape(-1, 8, 3)
# 计算SPFH
# pc_torch 1024*3, normal_torch 1024*3 neighborPoint 1024*8*3 neighborIdx 1024*8

uMat = torch.zeros(neighborPoint.shape) + normal_torch.view(-1, 1, 3)  # 1024 * 8 * 3
vMat = torch.linalg.cross(neighborPoint - pc_torch.view(-1, 1, 3), uMat)  # 1024 * 8 * 3
wMat = torch.linalg.cross(uMat, vMat) # 1024 * 8 * 3

alpha = torch.einsum("ijk,ijk->ij", vMat, neighborNormal)
PjPiDistance = torch.einsum("ijk->ij", torch.sqrt((neighborPoint - pc_torch.view(-1, 1, 3)) ** 2))
phi = torch.einsum("ijk,ijk->ij", uMat, neighborPoint - pc_torch.view(-1, 1, 3)) / PjPiDistance
theta = torch.arctan(torch.einsum("ijk,ijk->ij", wMat, neighborNormal)/
                     torch.einsum("ijk,ijk->ij", uMat, neighborNormal))

alpha_histogram = torch.histogram(alpha, bins=div, range=(-1.0, +1.0))[0]
phi_histogram = torch.histogram(phi, bins=div, range=(-1.0, +1.0))[0]
theta_histogram = torch.histogram(theta, bins=div, range=(-torch.pi, +torch.pi))[0]

alphaDiv = (alpha >= 0)
phiDiv = (phi >= 0)
thetaDiv = (theta >= 0)

histRes = torch.zeros((neighborIdx.shape[0], 8))
for i in range(neighborIdx.shape[0]):
    for j in range(neighborIdx.shape[1]):
        tmpPos = thetaDiv[i][j] * 4 + phiDiv[i][j] * 2 + alphaDiv[i][j]
        histRes[i][tmpPos] += 1

print(histRes-torch.from_numpy(spfhHist).float())
print(torch.allclose(histRes,torch.from_numpy(spfhHist).float()))
