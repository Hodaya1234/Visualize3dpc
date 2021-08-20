import numpy as np
from utils.activations_and_gradients import ActivationsAndGradients


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.device=device
        self.model = model.eval()
        self.target_layer = target_layer
        self.model = model.to(device)

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input):
        return self.model(input)

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_weights(self, grads): # grads= BxCxNs, out= BxC
        return np.mean(grads, axis=2)

    def get_cam_image(self, input_tensor, target_category,activations, grads):
        weights = self.get_cam_weights(grads)  # weights= BxC =Bx1024, activations=BxCxNs =Bx1024x1024
        weighted_activations = activations * weights[:,:, None]

        cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None):

        input_tensor.to(self.device)
        output = self.activations_and_grads(input_tensor)# Forward pass - create activation
        self.model.zero_grad()

        loss = self.get_loss(output, target_category)# loss=output[Target Category]
        loss.backward(retain_graph=True)# Backward pass - create gradients

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        cam = self.get_cam_image(input_tensor, target_category, activations, grads)
        cam = np.maximum(cam, 0)  # Relu

        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / np.max(img)
            result.append(img)
            result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        return self.forward(input_tensor, target_category)
