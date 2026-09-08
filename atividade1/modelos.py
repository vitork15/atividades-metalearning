import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights, vgg11, VGG11_Weights, resnet18, ResNet18_Weights, inception_v3, Inception_V3_Weights, efficientnet_b2, EfficientNet_B2_Weights

class PretrainAlexNet(nn.Module):
    '''
    AlexNet pretreinada no dataset ImageNet-1K com preprocessamento embutido. O parâmetro `freeze` congela a rede inteira exceto pela última camada.
    '''
    
    def __init__(self, num_classes, freeze=True):
        super().__init__()

        self.model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.transform = AlexNet_Weights.IMAGENET1K_V1.transforms()

        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad,self.parameters())
    
class PretrainVGG11(nn.Module):
    '''
    VGG11 pretreinada no dataset ImageNet-1K com preprocessamento embutido. O parâmetro `freeze` congela a rede inteira exceto pela última camada.
    '''
    
    def __init__(self, num_classes, freeze=True):
        super().__init__()

        self.model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        self.transform = VGG11_Weights.IMAGENET1K_V1.transforms()

        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad,self.parameters())
    
class PretrainResNet18(nn.Module):
    '''
    ResNet18 pretreinada no dataset ImageNet-1K com preprocessamento embutido. O parâmetro `freeze` congela a rede inteira exceto pela última camada.
    '''
    
    def __init__(self, num_classes, freeze=True):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad,self.parameters())
    
class PretrainInceptionV3(nn.Module):
    '''
    InceptionV3 pretreinada no dataset ImageNet-1K com preprocessamento embutido. O parâmetro `freeze` congela a rede inteira exceto pela última camada.
    '''
    
    def __init__(self, num_classes, freeze=True):
        super().__init__()

        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.transform = Inception_V3_Weights.IMAGENET1K_V1.transforms()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        if self.model.training:
            return self.model(x).logits # ignorei a loss auxiliar para o finetune
        else:
            return self.model(x)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad,self.parameters())
    
class PretrainEfficientNetB2(nn.Module):
    '''
    EfficientNetB2 pretreinada no dataset ImageNet-1K com preprocessamento embutido. O parâmetro `freeze` congela a rede inteira exceto pela última camada.
    '''
    
    def __init__(self, num_classes, freeze=True):
        super().__init__()

        self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.transform = EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()

        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad,self.parameters())