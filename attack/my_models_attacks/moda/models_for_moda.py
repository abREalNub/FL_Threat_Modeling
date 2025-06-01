import torch.nn as nn
import torch
import torch.nn.functional as F


""""" Esto importante para obtener las dimensiones de cualquier modelo
import torch
from torchvision.models import vgg11_bn

# Crear un modelo VGG11_bn
model = vgg11_bn()

# Crear un tensor de entrada ficticio
input_tensor = torch.randn(1, 3, 224, 224)

# Pasar el tensor a través de las capas de características (features)
features = model.features(input_tensor)

# Verificar las dimensiones de salida
print(features.shape)  # Debería imprimir torch.Size([1, 512, 7, 7])
"""


def get_models(data_name, base_model_D, n_class, dim_input):
    D = Custom_model(base_model_D, dim_input, n_class)
    latent_size = 128#Para los generadores, definirl la dimensión de entrada
    G = None
    if data_name == "MNIST" or data_name == "FASHIONMNIST" or data_name == "EMNIST":
        G = GeneratorMinist(latent_size, n_class)
    else:
        G = GeneratorCifar(latent_size, latent_size, n_class)
    return D, G


class Custom_model(nn.Module):
   def __init__(self, base_model, feature_dim, num_classes):
    super(Custom_model, self).__init__()
    self.base_model = base_model["model"]

    fict_data = torch.randn(1, feature_dim[0], feature_dim[1], feature_dim[2])
    features = self.base_model.features(fict_data)
    flattened = features #features.view(features.size(0), -1)

    self.l_gan_logit = nn.Linear(flattened, 1)
    self.classifier = nn.Linear(flattened, num_classes)
    
   def forward(self, x):
    # Pasar los datos a través del modelo base
    num_layers = sum(1 for _,_ in self.base_model.named_children())
    print(num_layers)
    i=0
    for n_m, module in self.base_model.named_children():
        #print(n_m)
        #x = module(x)
        i+=1
        if i == num_layers:
            break
        #print(n_m)
        x = module(x)
       # print(type(x))
#        else:
#           print(n_m)  
#            x = module(x)       
    #x = self.base_model(x)
    x = x.view(x.size(0), -1)  # Aplanar los tensores
        
    # Pasar por las capas del discriminador
    logit_gan = self.l_gan_logit(x)
    class_output = self.classifier(x)
        
    return torch.sigmoid(logit_gan).squeeze(1), class_output




class GeneratorMinist(nn.Module):
    def __init__(self, latent_size, nb_classes):
        super(GeneratorMinist, self).__init__()
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
        
        self.conv_aux = nn.Conv2d(3, 1, kernel_size=1)

        self.resize_aux = nn.Upsample(size=(28, 28), mode='bilinear')

    def forward(self, input, cl):
        x = torch.mul(self.label_embedding(cl.long()), input)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        #Adicionar esto solo si hay que hacer cambios en la imagen, se deben pasar los canales y la nueva dimensión
        x = self.conv_aux(x)
        x = self.resize_aux(x)

        return torch.tanh(x)







class GeneratorCifar(torch.nn.Module):
    def __init__(self, z_dim, dim_g, nb_classes, use_bn=True, **kw):
        super(GeneratorCifar, self).__init__(**kw)
        self.dim_g = dim_g

        self.label_embedding = nn.Embedding(nb_classes, 128)
        self.linear = nn.Linear(z_dim, 4*4*dim_g)
        self.res1 = ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
        self.res2 = ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
        self.res3 = ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn)
        self.normal = Normalize(dim_g)
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

class ResidualBlock(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, resample=None, use_bn=False, **kw):
        super(ResidualBlock, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        bn2dim = outdim
        if resample == "down":
            self.conv1 = torch.nn.Conv2d(indim, indim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv2 = ConvMeanPool(indim, outdim, filter_size=filter_size)
            self.conv_shortcut = ConvMeanPool
            bn2dim = indim
        elif resample == "up":
            self.conv1 = UpsampleConv(indim, outdim, filter_size=filter_size)
            self.conv2 = torch.nn.Conv2d(outdim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv_shortcut = UpsampleConv
        else:   # None
            assert(resample is None)
            self.conv1 = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv2 = torch.nn.Conv2d(outdim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv_shortcut = torch.nn.Conv2d
        if use_bn:
            self.bn1 = Normalize(indim)
            self.bn2 = Normalize(bn2dim)
        else:
            self.bn1, self.bn2 = None, None

        self.nonlin = torch.nn.ReLU()

        if indim == outdim and resample == None:
            self.conv_shortcut = None
        else:
            self.conv_shortcut = self.conv_shortcut(indim, outdim, filter_size=1)       # bias is True by default, padding is 0 by default

    def forward(self, x):
        if self.conv_shortcut is None:
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
        y = self.bn1(x) if self.bn1 is not None else x
        y = self.nonlin(y)
        y = self.conv1(y)
        y = self.bn2(y) if self.bn2 is not None else y
        y = self.nonlin(y)
        y = self.conv2(y)

        return y + shortcut
    
class ConvMeanPool(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, biases=True, **kw):
        super(ConvMeanPool, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        self.conv = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=biases)
        self.pool = torch.nn.AvgPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        y = self.pool(y)
        return y
    
class UpsampleConv(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, biases=True, **kw):
        super(UpsampleConv, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        self.conv = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=biases)


    def forward(self, x):
        y = F.interpolate(x, scale_factor=2)
        y = self.conv(y)
        return y