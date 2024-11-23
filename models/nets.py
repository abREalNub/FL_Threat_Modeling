import torch
import torch as nn
from torchvision.models import resnet18, ResNet18_Weights, VGG11_Weights, vgg11_bn, VGG11_BN_Weights


def build_model(n_classes=10, dataname='mnist'):

    if dataname == 'cifar10' or dataname == 'cifar100':
        # model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # for name, param in model.named_parameters():
        #     if 'bn' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, n_classes)
        # model.fc.requires_grad = True

        # model = resnet18(num_classes=n_classes)

        model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        for param, layer_class in zip(model.features.parameters(), model.features):
            if type(layer_class) is nn.BatchNorm2d:
                param.requires_grad = True
            else:
                param.requires_grad = False

        num_ftrs = model._modules['classifier'][-1].in_features
        model._modules['classifier'][-1] = nn.Linear(
            num_ftrs, n_classes)

    else:
        model = CNN(n_classes)

    return model


class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64 * 2, kernel_size=3,
                      stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64 * 2, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64 * 4, kernel_size=3,
                      stride=2, padding=0, bias=True),
            nn.BatchNorm2d(64 * 4, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.out = nn.Linear(64 * 4 * 3 * 3, n_classes, bias=True)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.out(output)
        return output


'''
https://arxiv.org/abs/1511.06434
The first layer of the GAN, which takes a uniform noise distribution Z as input,
 could be called fully connected as it is just a matrix multiplication, 
 but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack. 
 For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output.
'''


class Discriminator(nn.Module):
    def __init__(self, disc_dim, no_of_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(no_of_channels, disc_dim, kernel_size=3,
                      stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(disc_dim, disc_dim * 2, kernel_size=3,
                      stride=2, padding=1, bias=True),
            nn.BatchNorm2d(disc_dim * 2, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(disc_dim * 2, disc_dim * 4, kernel_size=3,
                      stride=2, padding=0, bias=True),
            nn.BatchNorm2d(disc_dim * 4, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv4 = nn.Conv2d(disc_dim * 4, no_of_channels,
                               kernel_size=3, stride=1, padding=0, bias=False)
        self.final = nn.Sigmoid()

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.final(output)
        return output


class Generator(nn.Module):
    def __init__(self, noise_dim, gen_dim, no_of_channels):
        super(Generator, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_dim, out_channels=gen_dim*4,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(gen_dim*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=gen_dim*4, out_channels=gen_dim*2,
                               kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_dim*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=gen_dim*2, out_channels=gen_dim,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=gen_dim, out_channels=no_of_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv(x)
        return out
