from models.ARM_model import ARMLayer
from feature_visualization.feature_visualization import FeatureVisualization


class CarmClass(FeatureVisualization):
    def __init__(self):
        self.name = 'C-ARM'
        super(CarmClass, self).__init__()

    def calculate_features(self,feature_conv, weight, target):

        return feature_conv

    def load_new_model(self):
        classifier = ARMLayer(self.num_classes).to(self.device)
        return classifier

if __name__ == '__main__':
    trainer = CarmClass()
    trainer.eval()



