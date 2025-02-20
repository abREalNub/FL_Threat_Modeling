import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, VGG11_BN_Weights, vgg11_bn
import torch.nn.functional as F
from torchvision.models import alexnet, resnet18
import torch.nn.init as init

"""""
Complementaos para ejecutar

input_dim = X_train_tensor.shape[1]
hidden_dim = 100
output_dim = len(set(y))
model = NurseryNN(input_dim, hidden_dim, output_dim)

Entrenamiento de la red

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

la que voy a montar

class AdvancedNurseryNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(AdvancedNurseryNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
"""""


#Provisional
class NurseryNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NurseryNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.float()
        #print(x.dtype)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def build_model_for_tabular(in_put = 8, inner_put = 15,out_put = 4, dataname='nursery'):

    if dataname=='nursery':
        return NurseryNN(in_put, inner_put, out_put)
    
    pass

def build_model(n_classes=10, dataname='mnist'):

    if dataname=='nursery':
        return NurseryNN(8,15,4)
        

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
    
    def features(self, img_fict):
        with torch.no_grad():
            output = self.conv1(img_fict)
            output = self.conv2(output)
            output = self.conv3(output)
            output = output.view(output.size(0), -1)
            #output = self.out(output)
        return output.view(output.size(0), -1).size(1)
    

#Redes Gans
########################################################################################################################################################################################################
########################################################################################################################################################################################################

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
    


class modelsGangs():
    #Para minist
    #def __init__(self):
    #    pass    
    class Reshape(nn.Module):

        def __init__(self, *new_shape):
            super(modelsGangs.Reshape, self).__init__()
            self._new_shape = new_shape

        def forward(self, x):
            new_shape = (x.size(i) if self._new_shape[i] == 0 else self._new_shape[i] for i in range(len(self._new_shape)))
            return x.view(*new_shape)
    class NoOp(nn.Module):

        def __init__(self, *args, **keyword_args):
            super(modelsGangs.NoOp, self).__init__()

        def forward(self, x):
            return x

    def _get_norm_fn_2d(norm):  # 2d
        if norm == 'batch_norm':
            return nn.BatchNorm2d
        elif norm == 'instance_norm':
            return nn.InstanceNorm2d
        elif norm == 'none':
            return modelsGangs.NoOp
        else:
            raise NotImplementedError
    
    
            
    def _get_weight_norm_fn(weight_norm):
        if weight_norm == 'spectral_norm':
            return torch.nn.utils.spectral_norm
        elif weight_norm == 'weight_norm':
            return torch.nn.utils.weight_norm
        elif weight_norm == 'none':
            return modelsGangs.identity
        else:
            return NotImplementedError
    
    def identity(self, x, *args, **keyword_args):
        return x
    
    class DiscriminatorMinist(nn.Module):

        def __init__(self, x_dim, c_dim, dim=96, norm='none', weight_norm='spectral_norm'):
            super(modelsGangs.DiscriminatorMinist, self).__init__()

            norm_fn = modelsGangs._get_norm_fn_2d(norm)
            weight_norm_fn = modelsGangs._get_weight_norm_fn(weight_norm)

            def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
                return nn.Sequential(
                    weight_norm_fn(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)),
                    norm_fn(out_dim),
                    nn.LeakyReLU(0.2)
                )

            self.ls = nn.Sequential(  # (N, x_dim, 32, 32)
                conv_norm_lrelu(x_dim, dim),
                conv_norm_lrelu(dim, dim),
                conv_norm_lrelu(dim, dim, stride=2),  # (N, dim , 16, 16)

                conv_norm_lrelu(dim, dim * 2),
                conv_norm_lrelu(dim * 2, dim * 2),
                conv_norm_lrelu(dim * 2, dim * 2, stride=2),  # (N, dim*2, 8, 8)

                conv_norm_lrelu(dim * 2, dim * 2, kernel_size=3, stride=1, padding=0),
                conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),
                conv_norm_lrelu(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0),  # (N, dim*2, 6, 6)

                nn.AvgPool2d(kernel_size=6),  # (N, dim*2, 1, 1)
                modelsGangs.Reshape(-1, dim * 2),  # (N, dim*2)
            )

            self.l_gan_logit = weight_norm_fn(nn.Linear(dim * 2, 1))  # (N, 1)
            self.l_c_logit = nn.Linear(dim * 2, c_dim)  # (N, c_dim)

        def forward(self, x):
            # x: (N, x_dim, 32, 32)
            feat = self.ls(x)
            gan_logit = self.l_gan_logit(feat)
            l_c_logit = self.l_c_logit(feat)
            return torch.sigmoid(gan_logit).squeeze(1), l_c_logit
        
    class GeneratorMinist(nn.Module):
        def __init__(self, latent_size , nb_filter, nb_classes):
            super(modelsGangs.GeneratorMinist, self).__init__()
            self.label_embedding = nn.Embedding(nb_classes, latent_size)
            self.layer1 = nn.Sequential(nn.ConvTranspose2d(128,512,4,1,1,bias = False),
                                    nn.ReLU(True))

            #input 512*4*4
            self.layer2 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1,bias = False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))
            #input 256*8*8
            self.layer3 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1,bias = False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))
            #input 128*16*16
            self.layer4 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1,bias = False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))
            #input 64*32*32
            self.layer5 = nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1,bias = False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))

            self.layer6 = nn.Sequential(nn.ConvTranspose2d(32,3,1,1,0,bias = False))
        def forward(self, input, cl):
            x = torch.mul(self.label_embedding(cl.long()), input)
            x = x.view(x.size(0), -1, 1, 1)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            return torch.tanh(x)

        
    #Para Cifar100
    class AlexNetCifar10(nn.Module):
        def __init__(self, in_channels, num_classes, norm_type='bn', pretrained=False, imagenet=False):
            super(alexnet, self).__init__()

            params = []

            if num_classes == 1000 or imagenet:  # imagenet1000
                if pretrained:
                    norm_type = 'none'
                self.features = nn.Sequential(
                    self.ConvBlock(3, 64, 11, 4, 2, bn=norm_type),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    self.ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    self.ConvBlock(192, 384, 3, 1, 1, bn=norm_type),
                    self.ConvBlock(384, 256, 3, 1, 1, bn=norm_type),
                    self.ConvBlock(256, 256, 3, 1, 1, bn=norm_type),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.AdaptiveAvgPool2d((6, 6))
                )

                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_classes),
                )

                for layer in self.features:
                    if isinstance(layer, self.ConvBlock):
                        params.append(layer.conv.weight)
                        params.append(layer.conv.bias)

                for layer in self.classifier:
                    if isinstance(layer, nn.Linear):
                        params.append(layer.weight)
                        params.append(layer.bias)

                if pretrained:
                    self._load_pretrained_from_torch(params)
            else:
                self.features = nn.Sequential(
                    self.ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
                    nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                    self.ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                    nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                    self.ConvBlock(192, 384, bn=norm_type),
                    self.ConvBlock(384, 256, bn=norm_type),
                    self.ConvBlock(256, 256, bn=norm_type),
                    nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
                )
                self.l_gan_logit = nn.Linear(4 * 4 * 256, 1) 
                self.classifier = nn.Linear(4 * 4 * 256, num_classes)
        
        def _load_pretrained_from_torch(self, params):
            # load a pretrained alexnet from torchvision
            torchmodel = alexnet(True)
            torchparams = []
            for layer in torchmodel.features:
                if isinstance(layer, nn.Conv2d):
                    torchparams.append(layer.weight)
                    torchparams.append(layer.bias)

            for layer in torchmodel.classifier:
                if isinstance(layer, nn.Linear):
                    torchparams.append(layer.weight)
                    torchparams.append(layer.bias)

            for torchparam, param in zip(torchparams, params):
                assert torchparam.size() == param.size(), 'size not match'
                param.data.copy_(torchparam.data)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            aux = self.classifier(x)
            gan = self.l_gan_logit(x)
            return torch.sigmoid(gan).squeeze(1), aux
        
    class ConvBlock(nn.Module):
        def __init__(self, i, o, ks=3, s=1, pd=1, bn='bn', relu=True):
            super().__init__()

            self.conv = nn.Conv2d(i, o, ks, s, pd, bias=bn == 'none')

            if bn == 'bn':
                self.bn = nn.BatchNorm2d(o)
            elif bn == 'gn':
                self.bn = nn.GroupNorm(o // 16, o)
            elif bn == 'in':
                self.bn = nn.InstanceNorm2d(o)
            else:
                self.bn = None

            if relu:
                self.relu = nn.ReLU(inplace=True)
            else:
                self.relu = None

            self.reset_parameters()

        def reset_parameters(self):
            init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x
    
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
            super(modelsGangs.BasicBlock, self).__init__()

            self.convbnrelu_1 = self.ConvBlock(in_planes, planes, 3, stride, 1, bn=norm_type, relu=True)
            self.convbn_2 = self.ConvBlock(planes, planes, 3, 1, 1, bn=norm_type, relu=True)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = self.ConvBlock(in_planes, self.expansion * planes,
                                        1, stride, 0, bn=norm_type, relu=True)

        def forward(self, x):
            out = self.convbnrelu_1(x)
            out = self.convbn_2(out)
            out = out + self.shortcut(x)
            out = F.relu(out)
            return out


    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
            super(modelsGangs.Bottleneck, self).__init__()

            self.convbnrelu_1 = self.ConvBlock(in_planes, planes, 1, 1, 0, bn=norm_type, relu=True)
            self.convbnrelu_2 = self.ConvBlock(planes, planes, 3, stride, 1, bn=norm_type, relu=True)
            self.convbn_3 = self.ConvBlock(planes, self.expansion * planes, 1, 1, 0, bn=norm_type, relu=False)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = self.ConvBlock(in_planes, self.expansion * planes, 1, stride, 0, bn=norm_type, relu=False)

        def forward(self, x):
            out = self.convbnrelu_1(x)
            out = self.convbnrelu_2(out)
            out = self.convbn_3(out) + self.shortcut(x)
            out = F.relu(out)
            return out
    
    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10, norm_type='bn', pretrained=False, imagenet=False):
            super(modelsGangs.ResNet, self).__init__()
            self.in_planes = 64
            self.num_blocks = num_blocks
            self.norm_type = norm_type

            if num_classes == 1000 or imagenet:
                self.convbnrelu_1 = nn.Sequential(
                    self.ConvBlock(3, 64, 7, 2, 3, bn=norm_type, relu=True),  # 112
                    nn.MaxPool2d(3, 2, 1),  # 56
                )
            else:
                self.convbnrelu_1 = self.ConvBlock(3, 64, 3, 1, 1, bn=norm_type, relu=True)  # 32
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 32/ 56
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16/ 28
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8/ 14
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4/ 7
            self.linear = nn.Linear(512 * block.expansion, num_classes)
            self.l_gan_logit = nn.Linear(512 * block.expansion, 1) 

            if num_classes == 1000 and pretrained:
                assert sum(num_blocks) == 8, 'only implemented for resnet18'
                layers = [self.convbnrelu_1[0].conv, self.convbnrelu_1[0].bn]
                for blocklayers in [self.layer1, self.layer2, self.layer3, self.layer4]:
                    for blocklayer in blocklayers:
                        b1 = blocklayer.convbnrelu_1
                        b2 = blocklayer.convbn_2
                        b3 = blocklayer.shortcut
                        layers += [b1.conv, b1.bn, b2.conv, b2.bn]
                        if not isinstance(b3, nn.Sequential):
                            layers += [b3.conv, b3.bn]
                layers += [self.linear]

                self._load_pretrained_from_torch(layers)
            
        def _load_pretrained_from_torch(self, layers):
            # load a pretrained alexnet from torchvision
            torchmodel = resnet18(True)
            torchlayers = [torchmodel.conv1, torchmodel.bn1]
            for torchblocklayers in [torchmodel.layer1, torchmodel.layer2, torchmodel.layer3, torchmodel.layer4]:
                for blocklayer in torchblocklayers:
                    torchlayers += [blocklayer.conv1, blocklayer.bn1, blocklayer.conv2, blocklayer.bn2]
                    if blocklayer.downsample is not None:
                        torchlayers += [blocklayer.downsample[0], blocklayer.downsample[1]]

            for torchlayer, layer in zip(torchlayers, layers):
                assert torchlayer.weight.size() == layer.weight.size(), 'must be same'
                layer.load_state_dict(torchlayer.state_dict())
        
        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride, self.norm_type))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = self.convbnrelu_1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(out.size(0), -1)
            aux = self.linear(out)
            gan = self.l_gan_logit(out)

            return torch.sigmoid(gan).squeeze(1), aux
        
    class GeneratorCifar10(torch.nn.Module):
        def __init__(self, z_dim, dim_g, nb_classes, use_bn=True, **kw):
            super(modelsGangs.GeneratorCifar10, self).__init__(**kw)
            self.dim_g = dim_g

            self.label_embedding = nn.Embedding(nb_classes, 128)
            self.linear = nn.Linear(z_dim, 4*4*dim_g)
            self.res1 = self.ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
            self.res2 = self.ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
            self.res3 = self.ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
            self.normal = self.Normalize(dim_g)
            self.conv = torch.nn.Conv2d(dim_g, 3, kernel_size=3, padding=1)

        def forward(self, x, cl):
            x = torch.mul(self.label_embedding(cl), x)
            x = self.linear(x)
            x = x.view(x.size(0), self.dim_g, 4, 4)
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.normal(x)
            x = F.relu(x)
            x = self.conv(x)
            x = torch.tanh(x)
            return x
    
    class Normalize(torch.nn.Module):
        def __init__(self, dim, **kw):
            super(self.Normalize, self).__init__(**kw)
            self.bn = torch.nn.BatchNorm2d(dim)

        def forward(self, x):
            return self.bn(x)
    
    
    def ResNet9(self,**model_kwargs):
        return self.ResNet(self.BasicBlock, [1, 1, 1, 1], **model_kwargs)


    def ResNet18(self,**model_kwargs):
        return self.ResNet(self.BasicBlock, [2, 2, 2, 2], **model_kwargs)


    def ResNet34(self,**model_kwargs):
        return self.ResNet(self.BasicBlock, [3, 4, 6, 3], **model_kwargs)


    def ResNet50(self,**model_kwargs):
        return self.ResNet(self.Bottleneck, [3, 4, 6, 3], **model_kwargs)


    def ResNet101(self,**model_kwargs):
        return self.ResNet(self.Bottleneck, [3, 4, 23, 3], **model_kwargs)


    def ResNet152(self,**model_kwargs):
        return self.ResNet(self.Bottleneck, [3, 8, 36, 3], **model_kwargs)