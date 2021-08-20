import torch
from models.CAM_model import CAMLayer
import numpy as np
from feature_visualization.feature_visualization import FeatureVisualization


class CamClass(FeatureVisualization):
    def __init__(self):
        self.name = 'CAM'
        super(CamClass, self).__init__()
        self.weights = self.classifier.fc.weight.data.detach().cpu().numpy()

    def calculate_features(self,feature_conv, weight, target):
        B, Nc, Np = feature_conv.shape
        beforeDot = feature_conv.reshape((Nc, Np))
        cam = (weight[target].reshape(1, -1) @ beforeDot)
        cam = np.maximum(cam, 0)  # Relu
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        result = np.float32(cam)
        return result

    def load_new_model(self):
        classifier = CAMLayer(self.num_classes).to(self.device)
        return classifier


if __name__ == '__main__':
    trainer = CamClass()
    trainer.eval()
