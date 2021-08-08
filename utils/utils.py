import numpy as np
import torch

def class_name(num):
    names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
        'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door',
        'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
        'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
        'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa',
        'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
        'wardrobe', 'xbox']
    return names[num]

def get_neighbors(feat1, feat2):
    """
    For each element in feat1 find the nearest neighbor in feat2
    :param feat1:
    :param feat2:
    :return:
    """
    N,D = feat1.shape
    neighbors = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        distances = torch.mean((feat2 - feat1[i].view(1,-1))**2, dim=1)
        neighbors[i] = torch.argmin(distances)
    return neighbors

def get_neighbors_unique(feat1, feat2):
    """
    For each element in feat1 find the nearest neighbor in feat2
    :param feat1:
    :param feat2:
    :return:
    """
    N,D = feat1.shape
    neighbors = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        distances = torch.mean((feat2 - feat1[i].view(1,-1))**2, dim=1)
        distances[neighbors] *= 2
        neighbors[i] = torch.argmin(distances)
    return neighbors

def single_group_nearest_neighbor(points):
    N,D = points.shape
    neighbors = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        distances = torch.mean((points - points[i].view(1, -1)) ** 2, dim=1)
        distances[i] = 1000
        neighbors[i] = torch.argmin(distances)
    return neighbors

def init_points(N, method='3d_sphere'):
    """
    Initialize a point cloud
    :param N: number of points
    :param method:
        gaussian: randn around zero with std of 1
        zeros: all the points at zero
        plane: random on the x,y and zeros on z
    :return:
    """
    if method == 'gaussian':
        return np.random.randn(N,3)
    elif method == 'zeros':
        return np.zeros([N,3])
    elif method == 'plane':
        return np.concatenate([np.random.randn(N,2),np.zeros([N,1])],axis=1)
    elif method == '3d_sphere':
        points = np.random.randn(N,6)
        points[:, 0:3] = points[:, 0:3] / np.linalg.norm(points[:, 0:3], axis=1, keepdims=True)
        return points
    else:
        raise ValueError('Not a valid method for initializing points')
