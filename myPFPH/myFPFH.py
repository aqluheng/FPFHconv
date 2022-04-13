from matplotlib.pyplot import axis
import torch

import pfh.utils as utils
from pfh.pfh import PFH, SPFH, FPFH
from time import time


source_pc = utils.load_pc_np('data/plant_source.npy')

et, div, nneighbors, rad = 0.1, 2, 8, 0.03
Icp = SPFH(et, div, nneighbors, rad)   # Fast PFH
normS, indS = Icp.calc_normals(source_pc)
spfhHist = Icp.calcHistArray(source_pc, normS, indS)

Icp = FPFH(et, div, nneighbors, rad)   # Fast PFH
normS, indS = Icp.calc_normals(source_pc)
fpfhHist = Icp.calcHistArray(source_pc, normS, indS)

gt_spfhHist = torch.Tensor(spfhHist)
gt_fpfhHist = torch.Tensor(fpfhHist)


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

neighborPoint = pc_torch[neighborIdx.reshape(-1), :].reshape(-1, 8, 3)   # 计算邻居点
# Amat = torch.zeros(pc_torch.shape[0], pc_torch.shape[0])

Wmat = 1.0 / torch.linalg.norm(neighborPoint - pc_torch.view(-1, 1, 3), ord=2, axis=2)
Xmat = gt_spfhHist[neighborIdx.reshape(-1), :].reshape(-1, 8, 8)
spfh_neighborhood = 1.0 / nneighbors * torch.einsum("ij,ijk->ik", Wmat, Xmat)
fpfh = gt_spfhHist + spfh_neighborhood
fpfh = fpfh / torch.norm(fpfh)

AdjMat =  torch.zeros((pc_torch.shape[0], pc_torch.shape[0]))
Adjcondense = 1.0/ torch.sqrt(neighborPoint - pc_torch.view(-1, 1, 3)**2)
for i in range(len(Adjcondense)):
    for j, neighborP in enumerate(neighborIdx[i]):
        AdjMat[i][neighborP] = Adjcondense[i][j]

A = AdjMat / torch.sum(AdjMat,axis=1).view(-1,1) * nneighbors
A += torch.eye(pc_torch.shape[0]) * torch.sqrt(nneighbors)



print(fpfhHist-gt_fpfhHist)
print(Wmat.shape)
print(Xmat.shape)

# for i in range(Amat.shape[0]):
# Amat[i][i] = 1
