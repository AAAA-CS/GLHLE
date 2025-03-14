from scipy.special import softmax
import numpy as np
from faiss import normalize_L2
import scipy
import scipy.io as sio
import scipy.stats
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import slic


def tosegment(Dataset,n_segments=100):
    # 加载数据集
    if Dataset == 'UP':
        uPavia = sio.loadmat('./dataset/PaviaU.mat')
        data_hsi = uPavia['paviaU']
    elif Dataset == 'SV':
        SV = sio.loadmat('./dataset/Salinas_corrected.mat')
        data_hsi = SV['salinas_corrected']
    elif Dataset == 'KSC':
        KSC = sio.loadmat('./dataset/KSC.mat')
        data_hsi = KSC['KSC']
    elif Dataset == 'Houston':
        Houston = sio.loadmat('./dataset/Houston.mat')
        data_hsi = Houston['Houston']
    elif Dataset == 'Houston2018':
        Houston2018 = sio.loadmat('./dataset/Houston2018.mat')
        data_hsi = Houston2018['HoutonU2018_img']
    elif Dataset == 'WHU_Hi_HanChuan':
        WHU_Hi_HanChuan = sio.loadmat('./dataset/WHU_Hi_HanChuan.mat')
        data_hsi = WHU_Hi_HanChuan['WHU_Hi_HanChuan']
    else:
        raise ValueError("Unknown dataset")
    # SLIC超像素分割
    segments = slic(data_hsi, n_segments=n_segments, compactness=10)
    return segments



def heat_kernel_similarity(vec1, vec2, sigma):
    distance = np.linalg.norm(vec1 - vec2)
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

def normalize_L2(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norm

#超图
def one_iter_true(segments, train_indices,Z, _Y, k=20, max_iter=300, l2=True, alpha=0.99, classes=10):
    Z_cpu = Z.cpu().numpy()
    Z_cpu = np.ascontiguousarray(Z_cpu)
    _Y_cpu = _Y.cpu().numpy()
    if l2:
        Z_cpu = normalize_L2(Z_cpu)

    # 构造局部图
    H_local = np.zeros((len(train_indices), len(train_indices)))
    W_local = np.zeros(len(train_indices))  # 权重矩阵的对角线

    for i in range(len(train_indices)):
        for j in range(len(train_indices)):
            if segments[train_indices[i, 0], train_indices[i, 1]] == segments[train_indices[j, 0], train_indices[j, 1]]:
                H_local[i, j] = 1
                W_local[i] += heat_kernel_similarity(Z_cpu[i], Z_cpu[j], sigma=1)

    W_local = np.diag(W_local)  # 将权重矩阵变为对角矩阵

    # 超图度矩阵
    DV_local = np.diag(np.sum(H_local * W_local.diagonal(), axis=1))  # 按照定义计算 D_v
    DE_local = np.sum(H_local, axis=0)

    # 处理DE中的零值
    DE_local[DE_local == 0] = 1e-10

    DV_inv_local = np.diag(1.0 / np.sqrt(DV_local.diagonal()))
    DE_inv_local = np.diag(1.0 / DE_local)

    H_t_local = H_local.T
    L_local = DV_inv_local @ H_local @ W_local @ DE_inv_local @ H_t_local @ DV_inv_local  # 归一化拉普拉斯矩阵


    L_local = L_local - np.diag(np.diag(L_local))
    S_local = L_local.sum(axis=1)
    S_local[S_local == 0] = 1
    D_local = np.array(1. / np.sqrt(S_local))
    D_local = np.diag(D_local)
    L_local = D_local @ L_local @ D_local

    # 构造全局图，包含自身样本和前 k 个邻近样本
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='brute').fit(Z_cpu)
    distances, indices = nbrs.kneighbors(Z_cpu)
    sigma = 1
    W = np.zeros(Z_cpu.shape[0])  # 权重矩阵的对角线

    for i in range(Z_cpu.shape[0]):
        neighbors = indices[i, :]  # 包含自身样本和前 k 个邻近样本
        for j in neighbors:
            W[i] += np.exp(-np.linalg.norm(Z_cpu[i] - Z_cpu[j])**2 / (2 * sigma**2))

    W = np.diag(W)  # 将权重矩阵变为对角矩阵

    # 构建带权超图邻接矩阵 H
    H = np.zeros((Z_cpu.shape[0], Z_cpu.shape[0]))
    for i in range(Z_cpu.shape[0]):
        neighbors = indices[i, :]  # 包含自身样本和前 k 个邻近样本
        H[i, neighbors] = 1  # 构造超边

    # 超图的度矩阵
    DV = np.diag(np.sum(H * W.diagonal(), axis=1))  # 按照定义计算 D_v
    DE = np.sum(H, axis=0)

    # 处理 DE 中的零值
    DE[DE == 0] = 1e-10

    DV_inv = np.diag(1.0 / np.sqrt(DV.diagonal()))
    DE_inv = np.diag(1.0 / DE)

    H_t = H.T
    L_global = DV_inv @ H @ W @ DE_inv @ H_t @ DV_inv  # 归一化拉普拉斯矩阵


    L_global = L_global - np.diag(np.diag(L_global))
    S = L_global.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = np.diag(D)
    L_global = D @ L_global @ D


    L = (L_local + L_global)/2

    F = np.zeros((Z_cpu.shape[0], classes))
    A = np.eye(L.shape[0]) - alpha * L
    for i in range(classes):
        f, _ = scipy.sparse.linalg.cg(A, _Y_cpu[:, i], tol=1e-3, maxiter=max_iter)
        F[:, i] = f
    F[F < 0] = 0
    F = softmax(F, 1)

    return F





