##########################################################
# Feature Extractor
##########################################################

class FeatureExtractor:
    """This class facilitates extracting features from a pretrained network """

    def __init__(self, model):
        self.hook_handlers = []
        self.conv_fuse = []
        self.feature_1 = []
        self.pt_last = []
        self.model = model

    def save_conv_fuse(self, module, input, output):
        self.conv_fuse = output

    def save_pt_last(self, module, input, output):
        self.pt_last = output

    def save_features_1(self, module, input, output):
        self.feature_1 = output

    def _register_hooks(self):
        """
        Registers all the hooks to perform extraction.
        """
        self.hook_handlers.append(self.model.conv_fuse.register_forward_hook(self.save_conv_fuse))
        self.hook_handlers.append(self.model.pt_last.register_forward_hook(self.save_pt_last))
        #self.hook_handlers.append(self.model.gather_local_1.register_forward_hook(self.save_features_1))

    def _unregister_hooks(self):
        """
        Unregisters all the hooks after performing an extraction.
        """
        for handles in self.hook_handlers:
            handles.remove()

    def extract(self, batch):
        self._register_hooks()
        out = self.model(batch)
        self._unregister_hooks()
        return out
