import torch
from torch import nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stochastic(nn.Module):
    """Add randomness to the activation of a unit."""

    def __init__(self, noise_level=1.0):
        super().__init__()
        self.noise_level = noise_level

    def forward(self, x):
        noise = torch.distributions.normal.Normal(
            torch.zeros_like(x), scale=1
        ).rsample()
        return x + noise * self.noise_level


class VGG11(nn.Module):
    """VGG-11 model as described in the literature. With some extra convenience methods."""

    def __init__(self, num_channels=3, num_classes=200, classifier_size=4096):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = self._build_classifier(classifier_size, num_classes)
        self.initialize_weights()
        self.feature_readout_layers = [2, 6, 13, 20, 27]
        self.classifier_readout_layers = [1, 4, 7]
        self.readout_layer_names = [
            "conv1_relu",
            "conv2_relu",
            "conv3_relu",
            "conv4_relu",
            "conv5_relu",
            "fc1_relu",
            "fc2_relu",
            "word_relu",
        ]

    def _n_features(self):
        """Determine how many outputs are produced by the features pipeline."""
        size = 224
        n_channels = 3
        for m in self.features.modules():
            if isinstance(m, nn.MaxPool2d):
                size = size // 2
            elif isinstance(m, nn.Conv2d):
                n_channels = m.out_channels
        n_features = size * size * n_channels
        return n_features

    def _build_classifier(self, classifier_size, num_classes):
        return nn.Sequential(
            nn.Linear(self._n_features(), classifier_size),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, classifier_size),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, num_classes),
            nn.ReLU(inplace=False),  # <-- Only active in eval() mode
        )

    def get_sgd_params(self, args):
        return [
            {"params": self.features.parameters(), "lr": args.lr},
            {"params": self.classifier.parameters(), "lr": args.lr},
        ]

    def forward(self, X):
        out = self.features(X)
        out = torch.flatten(out, 1)
        if self.training:
            # Skip final module (ReLU)
            for i, module in enumerate(self.classifier):
                if i < (len(self.classifier) - 1):
                    out = module(out)
        else:
            out = self.classifier(out)
        return out

    def initialize_weights(self, features=True, classifier=True):
        if not features and not classifier:
            return
        elif features and not classifier:
            modules = self.features.modules()
        elif classifier and not features:
            modules = self.classifier.modules()
        else:
            modules = self.modules()
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_layer_activations(self, images, verbose=True):
        """Obtain activation of each readout layer on a set of images."""
        self.eval()
        with torch.no_grad():
            out = images
            for i, layer in enumerate(self.features):
                out = layer(out)
                if verbose:
                    print("feature layer %02d, output=%s" % (i, out.shape))
                if i in self.feature_readout_layers:
                    yield out.detach().cpu().numpy()
            out = torch.flatten(out, 1)
            for i, layer in enumerate(self.classifier):
                out = layer(out)
                if verbose:
                    print("classifier layer %02d, output=%s" % (i, out.shape))
                if i in self.classifier_readout_layers:
                    yield out.detach().cpu().numpy()

    def set_dropout(self, p):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def set_n_outputs(self, num_classes):
        output_layer = self.classifier[-2]
        prev_num_classes, classifier_size = output_layer.weight.shape
        if prev_num_classes == num_classes:
            print(f"=> not resizing output layer ({prev_num_classes} == {num_classes})")
            return
        print(f"=> resizing output layer ({prev_num_classes} => {num_classes})")
        output_layer = nn.Linear(classifier_size, num_classes)
        nn.init.normal_(output_layer.weight, 0, 0.01)
        nn.init.constant_(output_layer.bias, 0)
        self.classifier[-2] = output_layer

    def set_classifier_size(self, classifier_size):
        output_layer = self.classifier[-2]
        num_classes, prev_classifier_size = output_layer.weight.shape
        if prev_classifier_size == classifier_size:
            print(
                f"=> not resizing classifier layers ({prev_classifier_size} == {classifier_size})"
            )
            return
        print(
            f"=> resizing classifier layers ({prev_classifier_size} => {classifier_size})"
        )
        self.classifier = self._build_classifier(classifier_size, num_classes)
        self.initialize_weights(features=False, classifier=True)

    @classmethod
    def from_checkpoint(
        cls, checkpoint, num_classes=None, classifier_size=None, freeze=False
    ):
        """Construct this model from a stored checkpoint."""
        state_dict = checkpoint["state_dict"]
        features_state_dict = {
            k: v for k, v in state_dict.items() if k.startswith("features.")
        }
        classifier_state_dict = {
            k: v for k, v in state_dict.items() if k.startswith("classifier.")
        }
        num_channels = state_dict["features.0.weight"].shape[1]

        output_layer_weights = state_dict[
            sorted(k for k in classifier_state_dict.keys() if k.endswith(".weight"))[-1]
        ]
        prev_num_classes, prev_classifier_size = output_layer_weights.shape
        model = cls(num_channels, prev_num_classes, prev_classifier_size)
        if num_classes is not None:
            model.set_n_outputs(num_classes)
        if classifier_size is not None:
            model.set_classifier_size(classifier_size)

        # Load the features
        model.features.load_state_dict(
            {
                k: v
                for k, v in zip(
                    model.features.state_dict().keys(), features_state_dict.values()
                )
            },
            strict=True,
        )

        # Load the classifier layers if possible
        try:
            model.classifier.load_state_dict(
                {
                    k: v
                    for k, v in zip(
                        model.classifier.state_dict().keys(),
                        classifier_state_dict.values(),
                    )
                },
                strict=True,
            )
            print("All layers successfully initialized.")
        except RuntimeError:
            print("Classifier layers not initialized.")
            pass

        if freeze:
            if freeze == "first3conv":
                print("=> freezing first 3 conv layers")
                for layer in model.features[:14]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                print("=> freezing model")
                for layer in model.features:
                    for param in layer.parameters():
                        param.requires_grad = False
                for layer in model.classifier:
                    for param in layer.parameters():
                        param.requires_grad = False
        return model


class VGG11Stochastic(VGG11):
    """The VGG11 model with random unit activations.

    This is the network used in the final model of the paper.
    """

    def __init__(
        self, num_channels=3, num_classes=200, classifier_size=4096, noise_level=0.1
    ):
        super().__init__(num_channels, num_classes, classifier_size)

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.feature_readout_layers = [3, 8, 17, 26, 35]  # ReLu modules
        self.classifier_readout_layers = [2, 6, 10]
        self.initialize_weights()

    def _build_classifier(self, classifier_size, num_classes, noise_level=0.1):
        return nn.Sequential(
            nn.Linear(self._n_features(), classifier_size),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, classifier_size),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, num_classes),
            Stochastic(noise_level=0),
            nn.ReLU(inplace=False),  # <-- Only active in eval() mode
        )

    def set_n_outputs(self, num_classes):
        output_layer = self.classifier[-3]
        prev_num_classes, classifier_size = output_layer.weight.shape
        if prev_num_classes == num_classes:
            print(f"=> not resizing output layer ({prev_num_classes} == {num_classes})")
            return
        print(f"=> resizing output layer ({prev_num_classes} => {num_classes})")
        output_layer = nn.Linear(classifier_size, num_classes)
        nn.init.normal_(output_layer.weight, 0, 0.01)
        nn.init.constant_(output_layer.bias, 0)
        self.classifier[-3] = output_layer

    def set_classifier_size(self, classifier_size):
        output_layer = self.classifier[-3]
        num_classes, prev_classifier_size = output_layer.weight.shape
        if prev_classifier_size == classifier_size:
            print(
                f"=> not resizing classifier layers ({prev_classifier_size} == {classifier_size})"
            )
            return
        print(
            f"=> resizing classifier layers ({prev_classifier_size} => {classifier_size})"
        )
        self.classifier = self._build_classifier(classifier_size, num_classes)
        self.initialize_weights(features=False, classifier=True)


class VGG11NoBatchNorm(VGG11):
    """VGG-11 model as described in the literature without batch normalization."""

    def __init__(self, num_channels=3, num_classes=200, classifier_size=4096):
        super().__init__(num_channels, num_classes, classifier_size)

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.feature_readout_layers = [1, 4, 9, 14, 19]
        self.initialize_weights()


class VGG11L1Stochastic(VGG11Stochastic):
    """VGG-11 stochastic model, but with only a single classifier layer."""

    def __init__(
        self, num_channels=3, num_classes=200, classifier_size=4096, noise_level=0.1
    ):
        super().__init__(num_channels, num_classes, classifier_size, noise_level)
        self.classifier_readout_layers = [2]
        self.readout_layer_names = [
            "conv1_relu",
            "conv2_relu",
            "conv3_relu",
            "conv4_relu",
            "conv5_relu",
            "word_relu",
        ]

    def _build_classifier(self, classifier_size, num_classes, noise_level=0.1):
        return nn.Sequential(
            nn.Linear(self._n_features(), num_classes),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),  # <-- Only active in eval() mode
        )


class VGG11L2Stochastic(VGG11Stochastic):
    """VGG-11 stochastic model, but with only two classifier layers."""

    def __init__(
        self, num_channels=3, num_classes=200, classifier_size=4096, noise_level=0.1
    ):
        super().__init__(num_channels, num_classes, classifier_size, noise_level)
        self.classifier_readout_layers = [2, 6]
        self.readout_layer_names = [
            "conv1_relu",
            "conv2_relu",
            "conv3_relu",
            "conv4_relu",
            "conv5_relu",
            "fc1_relu",
            "word_relu",
        ]

    def _build_classifier(self, classifier_size, num_classes, noise_level=0.1):
        return nn.Sequential(
            nn.Linear(self._n_features(), classifier_size),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, num_classes),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),  # <-- Only active in eval() mode
        )


class VGG11L4Stochastic(VGG11Stochastic):
    """VGG-11 stochastic model, but with four classifier layers."""

    def __init__(
        self, num_channels=3, num_classes=200, classifier_size=4096, noise_level=0.1
    ):
        super().__init__(num_channels, num_classes, classifier_size, noise_level)
        self.classifier_readout_layers = [2, 6, 10, 14]
        self.readout_layer_names = [
            "conv1_relu",
            "conv2_relu",
            "conv3_relu",
            "conv4_relu",
            "conv5_relu",
            "fc1_relu",
            "fc2_relu",
            "fc3_relu",
            "word_relu",
        ]

    def _build_classifier(self, classifier_size, num_classes, noise_level=0.1):
        return nn.Sequential(
            nn.Linear(self._n_features(), classifier_size),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, classifier_size),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, classifier_size),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, num_classes),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),  # <-- Only active in eval() mode
        )


class VGG11C4Stochastic(VGG11Stochastic):
    """VGG-11 stochastic model, but with only 4 convolution layers."""

    def __init__(
        self, num_channels=3, num_classes=200, classifier_size=4096, noise_level=0.1
    ):
        super().__init__(num_channels, num_classes, classifier_size, noise_level)

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = self._build_classifier(
            classifier_size, num_classes, noise_level
        )
        self.feature_readout_layers = [3, 8, 17, 26]  # ReLu modules
        self.readout_layer_names = [
            "conv1_relu",
            "conv2_relu",
            "conv3_relu",
            "conv4_relu",
            "fc1_relu",
            "fc2_relu",
            "word_relu",
        ]
        self.initialize_weights()


class VGG11C3Stochastic(VGG11Stochastic):
    """VGG-11 stochastic model, but with only 3 convolution layers."""

    def __init__(
        self, num_channels=3, num_classes=200, classifier_size=4096, noise_level=0.1
    ):
        super().__init__(num_channels, num_classes, classifier_size, noise_level)

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = self._build_classifier(
            classifier_size, num_classes, noise_level
        )
        self.feature_readout_layers = [3, 8, 17]  # ReLu modules
        self.readout_layer_names = [
            "conv1_relu",
            "conv2_relu",
            "conv3_relu",
            "fc1_relu",
            "fc2_relu",
            "word_relu",
        ]
        self.initialize_weights()

    def _build_classifier(self, classifier_size, num_classes, noise_level=0.1):
        return nn.Sequential(
            nn.Linear(self._n_features(), classifier_size),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, classifier_size),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(classifier_size, num_classes),
            Stochastic(noise_level=noise_level),
            nn.ReLU(inplace=False),  # <-- Only active in eval() mode
        )


class VGG11C6Stochastic(VGG11Stochastic):
    """VGG-11 stochastic model, but with 6 convolution layers."""

    def __init__(
        self, num_channels=3, num_classes=200, classifier_size=4096, noise_level=0.1
    ):
        super().__init__(num_channels, num_classes, classifier_size, noise_level)
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Stochastic(noise_level=noise_level),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.feature_readout_layers = [3, 8, 17, 26, 35, 44]  # ReLu modules
        self.readout_layer_names = [
            "conv1_relu",
            "conv2_relu",
            "conv3_relu",
            "conv4_relu",
            "conv5_relu",
            "conv6_relu",
            "fc1_relu",
            "fc2_relu",
            "word_relu",
        ]
        self.classifier = self._build_classifier(
            classifier_size, num_classes, noise_level
        )
        self.initialize_weights()


vgg11 = VGG11
vgg11nobn = VGG11NoBatchNorm
vgg11stochastic = VGG11Stochastic
vgg11l1stochastic = VGG11L1Stochastic
vgg11l2stochastic = VGG11L2Stochastic
vgg11l4stochastic = VGG11L4Stochastic
vgg11c3stochastic = VGG11C3Stochastic
vgg11c4stochastic = VGG11C4Stochastic
vgg11c6stochastic = VGG11C6Stochastic
final = VGG11Stochastic  # the final model as described in the paper.
