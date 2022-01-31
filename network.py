import torch
from torch import nn


class VGG11(nn.Module):
    """VGG-11 model as described in the literature. With some extra convenience methods."""
    def __init__(self, num_channels=3, num_classes=200, classifier_size=4096):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, classifier_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_size, classifier_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_size, num_classes),
            nn.ReLU(True),  # <-- Added after training
        )

        self.initialize_weights()

    def forward(self, X):
        out = self.features(X)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_layer_activations(self, images,
                              feature_layers=[0, 4, 11, 18, 25],
                              classifier_layers=[0, 3, 6],
                              verbose=True):
        """Obtain activation of each layer on a set of images."""
        self.eval()
        batch_size = 600
        with torch.no_grad():
            out = images
            for i, layer in enumerate(self.features):
                layer_out = []
                for j in range(0, len(out), batch_size):
                    layer_out.append(layer(out[j:j + batch_size]))
                out = torch.cat(layer_out, 0)
                del layer_out
                if verbose:
                    print('feature layer %02d, output=%s' % (i, out.shape))
                if i in feature_layers:
                    yield out.detach().cpu().numpy().copy()
            out = out.view(out.size(0), -1)
            layer_out = []
            for i, layer in enumerate(self.classifier):
                layer_out = []
                for j in range(0, len(out), batch_size):
                    layer_out.append(layer(out[j:j + batch_size]))
                out = torch.cat(layer_out, 0)
                if verbose:
                    print('classifier layer %02d, output=%s' % (i, out.shape))
                if i in classifier_layers:
                    yield out.detach().cpu().numpy().copy()

    def set_n_outputs(self, num_classes):
        modulelist = list(self.classifier.modules())[1:]
        output_layer = modulelist[6]
        prev_num_classes, classifier_size = output_layer.weight.shape
        if prev_num_classes == num_classes:
            print(f'=> not resizing output layer ({prev_num_classes} == {num_classes})')
            return
        print(f'=> resizing output layer ({prev_num_classes} => {num_classes})')
        output_layer = nn.Linear(classifier_size, num_classes)
        nn.init.normal_(output_layer.weight, 0, 0.01)
        nn.init.constant_(output_layer.bias, 0)
        modulelist[6] = output_layer
        self.classifier = nn.Sequential(*modulelist)

    @classmethod
    def from_checkpoint(cls, checkpoint, num_classes=None, freeze=False):
        """Construct this model from a stored checkpoint."""
        state_dict = checkpoint['state_dict']
        num_channels = state_dict['features.0.weight'].shape[1]
        prev_num_classes = state_dict['classifier.6.weight'].shape[0]
        classifier_size = state_dict['classifier.6.weight'].shape[1]

        model = cls(num_channels, prev_num_classes, classifier_size)
        model.load_state_dict(state_dict)

        if num_classes is not None:
            model.set_n_outputs(num_classes)

        if freeze:
            print('=> freezing model')
            for layer in model.features:
                for param in layer.parameters():
                    param.requires_grad = False
            print('=> disabling tracking batchnorm running stats')
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        return model
