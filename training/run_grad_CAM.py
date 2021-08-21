import torch
from data.dataset import ModelNetDataLoader
from tqdm import tqdm

from grad_cam import GradCAM
from models.model import PointTransformerCls
import numpy as np
import os

def run():
    from config import config
    config = config.load_config('../config/config.ini')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = config['DataPrep']['dataset']
    original_model_path = os.path.join(config['Net']['original_model_path'],'best_model.pth')
    npoints = int(config['Net']['num_points'])
    output_path = os.path.join(config['Runtime']['output_path'], 'grad_CAM.npz')

    TEST_DATASET = ModelNetDataLoader(root=data_path, npoint=npoints, split='test', normal_channel=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)

    # Load model
    PCT_model = PointTransformerCls().to(device)
    PCT_checkpoint = torch.load(original_model_path, map_location=torch.device(device))
    PCT_model.load_state_dict(PCT_checkpoint['model_state_dict'])

    target_layer = PCT_model.conv_fuse
    cam = GradCAM(model=PCT_model, target_layer=target_layer, device=device)

    points_array = []
    target_array = []
    grad_CAM_array = []
    for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        points, target = data
        target = target[:, 0]
        points, target = points.to(device), target.to(device)
        grayscale_cam = cam(input_tensor=points, target_category=target)

        points_array.append(points.detach().cpu().numpy())
        target_array.append(target.detach().cpu().numpy())
        grad_CAM_array.append(grayscale_cam)

    np.savez(output_path, target=target_array, all_points=points_array, grad_CAM=grad_CAM_array)  # save all in one file

