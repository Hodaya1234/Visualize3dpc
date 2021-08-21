import numpy as np
import vis_functions
from utils import plot_points
import general_utils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    do_latent_single_clust = False
    do_latent_group_clust = False
    do_cam = False
    do_grad_cam = False
    do_arm = False
    do_pc_synt = True
    # Take the inputs
    reg_input = np.load('inputs/reg_input.npz')
    num_examples = 16
    js = [0,1,23,37,38,67,74,75,83,85,130]
    js = [0]#[23,37,38,67,74,75,83,85,130] #[75,83,134,135]
    for j in js:
        example_ind = np.arange(j*num_examples,(j+1)*num_examples)
        points = np.squeeze(reg_input['all_points'])[example_ind]
        preds_dist = np.squeeze(reg_input['pred'])[example_ind]
        preds = np.argmax(preds_dist, axis=1)
        targets = np.squeeze(reg_input['target'][example_ind])

        if do_pc_synt:
            pc = np.load('inputs/pc_synthesis/lamp_1194.npy')
            i=0
            plot_points.moving_points_video(pc, np.zeros([1,3]), title='results/synthesis/LAMP_2', sizes=2)
            plot_points.static_points_video(pc[-1], np.zeros([1,3]), sizes=2, n_frames=100,
                                            title='results/synthesis/LAMP_2_FINAL')

        if do_latent_single_clust or do_latent_group_clust:
            latent_file = np.load('inputs/latent_data.npz')
            latent_data = latent_file['latent'][example_ind]
            cluster_colors = vis_functions.get_latent_colors(latent_data, together=do_latent_group_clust, n_classes=3)
            title = 'results/cluster_single/{}_animation_{}' if not do_latent_group_clust else 'results/cluster_group/{}_animation_{}'
            for i in range(num_examples):
                plot_points.static_points_video(points[i], cluster_colors[i], sizes=4, n_frames=100, title=title.format(general_utils.class_name(targets[i]), j * num_examples + i))

        if do_cam:
            cam_file = np.load('inputs/CAM_data_new.npz')  # pred, target, all_points, latent, CAM, pts
            cam_data = np.squeeze(cam_file['CAM'][example_ind])
            cam_colors = vis_functions.get_grad_cam_colors(cam_data)
            for i in range(num_examples):
                plot_points.static_points_video(points[i], cam_colors[i], n_frames=100, title='results/cam/{}_animation_{}'.format(general_utils.class_name(targets[i]), j * num_examples + i))

        if do_grad_cam:
            grad_cam_file = np.load('inputs/grad_CAM_naive_PCT_89.npz')  # target, all_points, grad_CAM
            grad_cam_data = np.squeeze(grad_cam_file['grad_CAM'][example_ind])
            grad_cam_colors = vis_functions.get_grad_cam_colors(grad_cam_data)
            for i in range(num_examples):
                plot_points.static_points_video(points[i], grad_cam_colors[i], n_frames=100, title='results/grad_cam/{}_animation_{}'.format(general_utils.class_name(targets[i]), j * num_examples + i))

        if do_arm:
            arm_file = np.load('inputs/ARM_data_mean.npz',
                               allow_pickle=True)  # pred, target, all_points, latent, ARM, pts
            arm_data = arm_file['ARM'][j]
            assert (arm_file['target'][j] == targets).all()
            colors, sizes = vis_functions.get_arm_colors_sizes(arm_data, targets, preds)
            for i in range(num_examples):
                plot_points.static_points_video(points[i], colors[i], sizes=sizes[i], n_frames=100, title='results/arm/{}_animation_{}'.format(general_utils.class_name(targets[i]), j * num_examples + i))

