import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from utils import clustering
from scipy.special import softmax
import general_utils

def class_activation_maps(pc_features):
    """

    :param pc_features: [N,c] array of features
    :return: y_pred: [N,1] argmax prediction
    """
    y_pred = np.argmax(pc_features, axis=1)
    confidence = (pc_features[np.arange(len(y_pred)), y_pred] - pc_features.min(axis=1))
    return y_pred, confidence


def get_arm_colors_sizes(arm, targets, preds):
    n_examples = targets.shape[0]
    colors = np.zeros([n_examples, 1024, 4])
    sizes = np.zeros([n_examples, 1024])
    for i in range(n_examples):
        arm_i = arm[i]
        target_i = targets[i]
        pred_i = preds[i]

        pred_per_point, confidence = class_activation_maps(arm_i)
        confidence = (confidence / confidence.max()) * 10  # make the points a bit larger

        target_points = pred_per_point == target_i
        pred_points = (pred_per_point == pred_i) if target_i != pred_i else np.zeros_like(target_points)
        else_points = (pred_per_point != target_i) & (pred_per_point != pred_i)

        colors[i, target_points] = [0.35, 0.75, 0.4, 1]
        colors[i, else_points] = [0.3, 0.3, 0.3, 0.4]

        if target_i != pred_i:
            colors[i, pred_points] = [0.75, 0.3, 0.3, 1]
            sizes[i,pred_points] = 10
        sizes[i, target_points] = 15
        sizes[i, else_points] = 5
    return colors, sizes


def plot_arm(all_pcs, all_CAMs, all_targets, all_preds):
    all_preds_label = np.argmax(all_preds, axis=1)

    n_examples = all_pcs.shape[0]
    r = c = np.ceil(np.sqrt(n_examples))
    fig = plt.figure(figsize=(12.0, 8.0))

    for i in range(n_examples):
        points = all_pcs[i]
        cam = all_CAMs[i]
        target = all_targets[i]
        pred = all_preds_label[i]

        pred_per_point, confidence = class_activation_maps(cam)
        confidence = confidence / confidence.max()  # make the points a bit larger

        target_points = pred_per_point == target
        pred_points = (pred_per_point == pred) if target != pred else np.zeros_like(target_points)
        else_points = (pred_per_point != target) & (pred_per_point != pred)

        ax = fig.add_subplot(r, c, i + 1, projection='3d')
        ax.axis('off')

        # ax.scatter(all_total_points[:, 0], all_total_points[:, 1], all_total_points[:, 2], c='silver',s=1)

        ax.scatter(points[target_points, 0], points[target_points, 1], points[target_points, 2], c='darkgreen',
                   s=confidence[target_points],
                   label='target: {}'.format(general_utils.class_name(target)))
        ax.scatter(points[else_points, 0], points[else_points, 1], points[else_points, 2], c='dimgrey',
                   s=confidence[else_points])
        if target != pred:  # If the last legend need the handle to pred points for the red in the legend
            ax.scatter(points[pred_points, 0], points[pred_points, 1], points[pred_points, 2],
                       c='darkred', s=confidence[pred_points], label='predicted: {}'.format(general_utils.class_name(pred)))
        ax.legend()

    plt.show()


def plot_grad_cam(all_pcs, all_targets, colors):
    n_examples = all_pcs.shape[0]
    r = c = np.ceil(np.sqrt(n_examples))
    fig = plt.figure()

    for i in range(n_examples):
        points = all_pcs[i]
        target = np.squeeze(all_targets[i])

        ax = fig.add_subplot(r, c, i + 1, projection='3d')
        ax.axis('off')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors,
                   s=1,
                   label='target: {}'.format(general_utils.class_name(target)))

        ax.legend()

    plt.show()


def get_grad_cam_colors(cam):
    cmap = plt.get_cmap('jet')
    colors = cmap(cam)
    return colors


def get_latent_colors(all_latent, together=False, n_classes=3):
    B, N, D = all_latent.shape
    assert N == 1024
    assert D == 512
    # cmap = plt.get_cmap('inferno')
    color_options = np.zeros([4, 4])
    color_options[0] = [0.894, 0.102, 0.109, 1.0]
    color_options[1] = [0.31, 0.7 , 0.3 , 1.0]
    color_options[2] = [0.215, 0.494, 0.721, 1.0]
    color_options[3] = [0.0, 0.0, 0.0, 0.8]
    # color_options = cmap(np.linspace(start=0.2, stop=1, num=n_classes))
    # color_options = np.concatenate([color_options, np.zeros([1,4])], axis=0)
    # color_options[:,-1] = 0.8
    # add black color for outliers (if any)
    print('in latent colors')
    if together:
        flat_latent = all_latent.reshape([-1, D])
        pred_per_point = clustering.clustering(flat_latent, algorithm_type='SpectralClustering',
                                               algorythm_params={'n_clusters': n_classes})

        colors = color_options[pred_per_point]
        colors = colors.reshape([B, N, -1])
        return colors
    else:
        all_colors = np.zeros([B, N, 4])
        for i in range(B):
            pred_per_point = clustering.clustering(all_latent[i], algorithm_type='MiniBatchKMeans',
                                                   algorythm_params={'n_clusters': n_classes})
            colors = color_options[pred_per_point]
            all_colors[i] = colors
        return all_colors


def plot_latent(all_pcs, all_latent, all_targets, all_preds):
    all_preds_label = np.argmax(all_preds, axis=1)
    probabilities = softmax(all_preds, axis=1)
    n_examples = all_pcs.shape[0]
    n_examples = 2
    r = c = np.ceil(np.sqrt(n_examples))
    fig = plt.figure()

    for i in range(n_examples):
        points = all_pcs[i]
        latent = all_latent[i]
        # extractor.extract(points.view(1, -1, 6))
        # latent = extractor.conv_fuse
        target = all_targets[i]
        pred = all_preds_label[i]
        latent = latent.T  # N_pointx X N_features
        pred_per_point = clustering.clustering(latent, algorithm_type='SpectralClustering',
                                               algorythm_params={'n_clusters': 3})

        colors = np.array(list(islice(cycle(['#e41a1c', '#115a96', '#0fdb09', '#377eb8',
                                             '#a65628', '#984ea3', '#ff7f00', '#dede00',
                                             '#999999', '#f781bf', '#4daf4a']),
                                      int(max(pred_per_point) + 1))))
        colors = np.append(colors, ["#000000"])  # add black color for outliers (if any)
        colors = colors[pred_per_point]
        ax = fig.add_subplot(r, c, i + 1, projection='3d')
        ax.axis('off')
        # ax.scatter(all_total_points[:, 0], all_total_points[:, 1], all_total_points[:, 2], c='silver',s=1)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2)

        text = 'target {}: {:.3f}'.format(general_utils.class_name(target), probabilities[i, target])
        if pred != target:
            text += '\npred {}: {:.3f}'.format(general_utils.class_name(pred), probabilities[i, pred])
        ax.set_title(text)
    plt.show()


