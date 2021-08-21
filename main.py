from argparse import ArgumentParser
from training import compute_ARM, compute_CAM, run_grad_CAM
import numpy as np
from utils import vis_functions, general_utils, plot_points

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("retrain", help="re-train the model", action='store_true')
    parser.add_argument("recompute", help="re-compute the features", action='store_true')
    parser.add_argument("type", help="cam, arm, grad_cam, synth, latent_cluster", type=str)
    parser.add_argument("examples", type=int, nargs="+")
    args = parser.parse_args()

    examples = args['examples']
    num_examples = examples


    if args['type'] == 'cam':
        trainer = compute_CAM.CamClass()
        if args['retrain']:
            trainer.train()
        if args['recompute']:
            trainer.eval()
        cam_file = np.load('results/CAM.npz')  # pred, target, points, maps
        pred = cam_file['pred'][examples]
        target = cam_file['target'][examples]
        points = cam_file['points'][examples]
        maps = cam_file['maps'][examples]

        cam_colors = vis_functions.get_grad_cam_colors(maps)
        for i, ind in enumerate(examples):
            plot_points.static_points_video(points[i], cam_colors[i],  n_frames=100,
                                            title='results/arm/{}_animation_{}'.format(
                                                general_utils.class_name(target[i]), ind))

    elif args['type'] == 'arm':
        trainer = compute_ARM.CarmClass()
        if args['retrain']:
            trainer.train()
        if args['recompute']:
            trainer.eval()
        arm_file = np.load('results/C-ARM.npz')  # pred, target, points, maps
        pred = arm_file['pred'][examples]
        target = arm_file['target'][examples]
        points = arm_file['points'][examples]
        maps = arm_file['maps'][examples]
        colors, sizes = vis_functions.get_arm_colors_sizes(maps, target, pred)
        for i, ind in enumerate(examples):
            plot_points.static_points_video(points[i], colors[i], sizes=sizes[i], n_frames=100,
                                            title='results/arm/{}_animation_{}'.format(
                                                general_utils.class_name(target[i]), ind))


    elif args['type'] == 'grad_cam':
        if args['recompute']:
            run_grad_CAM.run()
        grad_cam_file = np.load('results/grad_CAM.npz') # target, all_points, grad_CAM
        grad_CAM = grad_cam_file['grad_CAM'][examples]
        target = grad_cam_file['target'][examples]
        points = grad_cam_file['all_points'][examples]
        grad_cam_colors = vis_functions.get_grad_cam_colors(grad_CAM)
        for i, ind in enumerate(examples):
            plot_points.static_points_video(points[i], grad_cam_colors[i], n_frames=100,
                                            title='results/grad_cam/{}_animation_{}'.format(
                                                general_utils.class_name(target[i]), ind))

    # elif args['type'] == 'synth':
    #     pc = np.load('inputs/pc_synthesis/lamp_1194.npy')
    #     i = 0
    #     plot_points.moving_points_video(pc, np.zeros([1, 3]), title='results/synthesis/LAMP_2', sizes=2)
    #     plot_points.static_points_video(pc[-1], np.zeros([1, 3]), sizes=2, n_frames=100,
    #                                     title='results/synthesis/LAMP_2_FINAL')
    #
    # elif args['type'] == 'latent_cluster':
    #     latent_file = np.load('inputs/latent_data.npz')
    #     latent_data = latent_file['latent'][examples]
    #     cluster_colors = vis_functions.get_latent_colors(latent_data, together=do_latent_group_clust, n_classes=3)
    #     title = 'results/cluster_single/{}_animation_{}' if not do_latent_group_clust else 'results/cluster_group/{}_animation_{}'
    #     for i in range(num_examples):
    #         plot_points.static_points_video(points[i], cluster_colors[i], sizes=4, n_frames=100,
    #                                         title=title.format(general_utils.class_name(targets[i]),
    #                                                            j * num_examples + i))

    else:
        raise NotImplementedError



