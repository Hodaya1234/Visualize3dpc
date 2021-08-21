import hydra
import numpy as np
import torch
import sys
from models import model
from utils.general_utils import init_points
from utils.feature_extractor import FeatureExtractor
from utils import plot_points
from scipy.special import softmax
# from attention import simple_self_attention
import utils
import os

def attention_loss(feat1, feat2):
    # B,C,N
    norm1 = feat1 - feat1.mean(dim=1, keepdims=True)/feat1.std(dim=1, keepdims=True)
    norm2 = feat2 - feat2.mean(dim=1, keepdims=True)/feat2.std(dim=1, keepdims=True)
    mat1 = norm1.permute(0, 2, 1) @ norm1 # B,N,N
    mat2 = norm2.permute(0, 2, 1) @ norm2 # B,N,N

    # att1 = simple_self_attention(feat1)
    # att2 = simple_self_attention(feat2)
    return torch.mean((mat1 - mat2)**2)


def mean_max_loss(feat1, feat2):
    mean_loss = torch.mean((feat1.mean(-1) - feat2.mean(-1))**2)
    max_loss = torch.mean((feat1.max(-1).values - feat2.max(-1).values) ** 2)
    return mean_loss + max_loss


def diff_loss(feat1, feat2):
    return ((feat1-feat2)**2).mean(dim=1).mean()


def nearest_neighbor_loss(feat1, feat2):
    neighbors = utils.get_neighbors_unique(feat1.detach(), feat2.detach())
    print(len(torch.unique(neighbors)))
    loss = ((feat1 - feat2[neighbors])**2).mean(dim=1).mean()
    return loss


def nearest_neighbor_reg(point_cloud, features):
    neighbors = utils.single_group_nearest_neighbor(point_cloud)
    reg = ((features - features[neighbors])**2).mean(dim=1).mean()
    return reg

def l2_reg(points):
    return torch.mean(torch.norm(points[:, 0:3], dim=1))


def optimize_pc(extractor, target_pc, input_pc, lr=1e-2, epochs=1000, step_size=750, gamma=0.5, title=''):
    # Params
    epochs = epochs
    lr = lr

    optimizer = torch.optim.Adam([input_pc], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nearest_neighbor_loss

    animator = plot_points.PointsAnimator(dir='results', title=title)

    plot_points.plot_3dpc(target_pc[:, :3], size=2)
    animator.add_frame(input_pc[:, 0:3].detach().cpu().numpy())

    extractor.extract(target_pc.view(1, -1, 6))
    target_out = extractor.conv_fuse # B,C,N

    # Optimize
    all_synth = np.zeros([epochs, 1024,3])
    for epoch in range(epochs):
        optimizer.zero_grad()
        extractor.extract(input_pc.view(1, -1, 6))
        synt_out = extractor.conv_fuse # B,C,N

        pc_loss = criterion(target_out.squeeze().T, synt_out.squeeze().T) + criterion(synt_out.squeeze().T, target_out.squeeze().T)

        print('Epoch %d loss: %.3f' % (epoch+1, pc_loss))
        pc_loss.backward()
        optimizer.step()
        scheduler.step()
        all_synth[epoch] = input_pc[:, 0:3].detach().cpu().numpy()
        animator.add_frame(input_pc[:, 0:3].detach().cpu().numpy())

    np.save('inputs/pc_synthesis/' + title, all_synth)
    animator.make_animation()


def run():
    from config import config
    config = config.load_config('../config/config.ini')
    lr = config['Synth']['lr']
    epochs = config['Synth']['epochs']
    step_size = config['Synth']['step_size']
    gamma = config['Synth']['gamma']
    # Params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = 'synth'
    output_path = os.path.join(config['Runtime']['output_path'], name + '.npz')

    # load PCT network
    PCT_model = model()
    if torch.cuda.is_available():
        PCT_checkpoint = torch.load('results/best_model.pth')
    else:
        PCT_checkpoint = torch.load('results/best_model.pth', map_location=torch.device('cpu'))

    PCT_model.load_state_dict(PCT_checkpoint['model_state_dict'])

    PCT_model.eval()
    for param in PCT_model.parameters():
        param.requires_grad_(False)
    PCT_model.to(device)

    extractor = FeatureExtractor(PCT_model)

    # Take an original point cloud
    # Target latent
    data_path = 'inputs/reg_input.npz'
    data = np.load(data_path, allow_pickle=True)
    # j = 0
    # i = 2
    # k = j*16 + i
    k = 1194
    original_pc = torch.from_numpy(data['all_points'][k]).to(device).float().squeeze()
    targets = np.squeeze(data['target'])

    N = len(original_pc)
    pc_synt = torch.from_numpy(init_points(N)).float().to(device).requires_grad_(True)

    title = 'inputs/pc_synthesis/{}_{}'.format(utils.class_name(targets[k]), k)
    optimize_pc(extractor, original_pc, pc_synt, lr=lr, epochs=epochs, step_size=step_size, gamma=gamma, title=title)

if __name__ == '__main__':
    run()