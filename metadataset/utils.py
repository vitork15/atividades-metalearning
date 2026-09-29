import numpy as np
import torch
from functools import partial
from scipy.stats import kurtosis, skew
from sklearn.metrics import mutual_info_score
from modelos import *
from torchvision import transforms
from pymfe.mfe import MFE
from torchmetrics.image import TotalVariation
from torchmetrics.image.arniqa import ARNIQA
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import Subset
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

from torchvision.datasets import (
    CIFAR10, CIFAR100, FashionMNIST, SVHN, STL10, DTD,
    Caltech101, Caltech256, EuroSAT, GTSRB,
    OxfordIIITPet, Food101
)

def get_num_classes(dataset):       
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    elif hasattr(dataset, "labels"):
        return len(set(dataset.labels))
    elif hasattr(dataset, "categories"):
        return len(set(dataset.categories))
    elif hasattr(dataset, "targets"):
        return len(np.unique(dataset.targets))
    elif isinstance(dataset, GTSRB):
        return 43

def get_labels(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)

    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)

    if hasattr(dataset, "_labels"):
        return np.asarray(dataset._labels)

    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    return np.asarray(labels)

def get_datasets():
    datasets = [('CIFAR10',partial(CIFAR10,root="./datasets",train=True,download=True)), 
                ('CIFAR100',partial(CIFAR100,root="./datasets",train=True,download=True)),
                ('FashionMNIST',partial(FashionMNIST,root="./datasets",train=True,download=True)),
                ('SVHN',partial(SVHN,root="./datasets",split='train',download=True)),
                ('STL10',partial(STL10,root="./datasets",split='train',download=True)),
                ('DTD',partial(DTD,root="./datasets",split='train',download=True)),
                ('Caltech-101',partial(Caltech101,root="./datasets",target_type='category',download=True)),
                ('Caltech-256',partial(Caltech256,root="./datasets",download=True)),
                ('EuroSAT',partial(EuroSAT,root="./datasets",download=True)),
                ('GTSRB',partial(GTSRB,root="./datasets",split='train',download=True)),
                #download lento ('Stanford Cars',partial(StanfordCars,root="./datasets",split='train',download=True)),
                #download lento ('FGVC Aircraft',partial(FGVCAircraft,root="./datasets",split='trainval',download=True)),
                #('Oxford-IIIT Pet',partial(OxfordIIITPet,root="./datasets",split='trainval',download=True)),
                #download lento ('Oxford Flowers102',partial(Flowers102,root="./datasets",split='train',download=True)),
                ('Food-101',partial(Food101,root="./datasets",split='train',download=True))
                #muito grande ('SUN397',partial(SUN397,root="./datasets",download=True)),
                #muito grande ('Places365',partial(Places365,root="./datasets",split='train-standard',small=True,download=True))
                ]  
    return datasets

def get_models():
    models = [('AlexNet-1L', PretrainAlexNet), 
            # muito pesado por conta das camadas FC ('VGG11-1L', PretrainVGG11), 
            ('ResNet18-1L', PretrainResNet18),
            ('Inception-1L', PretrainInceptionV3),
            # lento de treinar ('EfficientNetB2-1L', PretrainEfficientNetB2),
            ('AlexNet-NF', partial(PretrainAlexNet, freeze=False)), 
            #muito pesado por conta das camadas FC ('VGG11-NF', partial(PretrainVGG11, freeze=False)), 
            ('ResNet18-NF', partial(PretrainResNet18, freeze=False)),
            # lento de treinar ('Inception-NF', partial(PretrainInceptionV3, freeze=False))
            # extremamente lento de treinar ('EfficientNetB2-NF', partial(PretrainEfficientNetB2, freeze=False))
            ]
    return models

def image_to_numpy(image):
    """
    Converte uma imagem PIL em um np.ndarray
    """

    image = np.asarray(image)
    image = image.astype(np.float32)
    image = image/255.0

    return image

def get_dataset_metafeatures(dataset, extended=False, subsample=False):
    """
    Calcula as meta-características de um dataset e retorna um dicionário `dict` com as meta-características.
    """

    n_samples = len(dataset)
    n_classes = get_num_classes(dataset)
    labels = get_labels(dataset)

    first_image, _ = dataset[0]
    first_image = image_to_numpy(first_image)

    height, width, channels = first_image.shape

    channel_sum = np.zeros(channels)
    channel_sum_sq = np.zeros(channels)
    
    total_pixels = 0

    for image, _ in dataset:

        image = image_to_numpy(image)

        pixels = image.reshape(-1, channels)
        
        total_pixels += pixels.shape[0]
        
        channel_sum += pixels.sum(axis=0)
        channel_sum_sq += np.sum(pixels ** 2, axis=0)
    
    mean = channel_sum / total_pixels
    variance = (channel_sum_sq/total_pixels) - mean**2 # E[X**2]-E[X]**2
    std = np.sqrt(variance)

    class_counts = np.bincount(labels)
    probabilities = class_counts[class_counts > 0] / n_samples
    class_entropy = -np.sum(probabilities * np.log2(probabilities))
    
    metafeature_dict = {
        "num_samples": n_samples,
        "num_classes": n_classes,

        "area": height*width,

        "pixel_mean": mean.tolist(),
        "pixel_std": std.tolist(),

        "class_entropy": float(class_entropy)
    }
    
    if extended:
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # define o extrator de features usando o corpo do vgg19 pretreinado
        vgg = vgg19(weights='IMAGENET1K_V1') 
        feature_extractor = nn.Sequential(vgg.features, nn.Flatten())
        feature_extractor.to(device=device)
        feature_extractor.eval()
        
        # subsample p/ reduzir custo de memoria da operação (praticamente inviavel sem subsampling)
        max_samples = 5000 
        if subsample and len(dataset) > max_samples:
            subsample_idx, _ = train_test_split(
                np.arange(len(dataset)),
                train_size=max_samples,
                stratify=labels,
                random_state=1234
            )
            sampled_dataset = Subset(dataset, subsample_idx)
        else:
            sampled_dataset = dataset

        sampled_labels = [sampled_dataset[i][1] for i in range(len(sampled_dataset))]
        sampled_size = len(sampled_dataset)
        
        # extratores de indices de qualidade
        clip_extractor = CLIPImageQualityAssessment().to(device)
        arniqa_extractor = ARNIQA().to(device)
        tv_extractor = TotalVariation(reduction='none').to(device)
        
        clip_extractor.eval()
        arniqa_extractor.eval()
        tv_extractor.eval()
        
        sampled_features = []
        clip = []
        arniqa = []
        tv = []
        
        quality_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        vgg_transform = VGG19_Weights.IMAGENET1K_V1.transforms()
        
        starting_transform = deepcopy(dataset.transform)
        
        with torch.no_grad():
            
            dataset.transform = quality_transform if starting_transform is None else transforms.Compose([starting_transform, quality_transform])
            
            # primeiro loop extrai as caracteristicas de qualidade usando um resize+recrop similar ao pre-processamento da resnet
            for image, _ in DataLoader(sampled_dataset, batch_size=32):
                image = image.to(device)
                tv.append(tv_extractor(image).cpu().numpy())
                arniqa.append(image.shape[0]*arniqa_extractor(image).cpu().numpy()) # arniqa nao funciona com reduction='none', multiplica o batch pela media obtida
                clip.append(clip_extractor(image).cpu().numpy())
                
            dataset.transform = vgg_transform if starting_transform is None else transforms.Compose([starting_transform, vgg_transform])
        
            # segundo loop extrai os embeddings de imagem usando o vgg19 com o pre-processamento padrao do vgg
            for image, _ in DataLoader(sampled_dataset, batch_size=32):
                image = image.to(device)
                sampled_features.append(feature_extractor(image).cpu().numpy())

        # reduzindo a dimensionalidade do embedding usando PCA (num de componentes baseado no artigo referencia)
        sampled_features = np.concatenate(sampled_features, axis=0)
        pca = PCA(n_components=154, random_state=1234)
        reduced_features = pca.fit_transform(sampled_features) 
        
        avg_clip_iqa = np.mean(np.concatenate(clip, axis=0))
        avg_arniqa = np.sum(arniqa, axis=0)/sampled_size
        avg_tv = np.mean(np.concatenate(tv, axis=0))
        
        quality_features = {
            "tv":avg_tv,
            "arniqa":avg_arniqa,
            "clip_iqa":avg_clip_iqa
        }
        
        # extração das metafeatures de complexidade
        feature_names = ['l1', 'l2', 'l3', 't1', 'n1', 'f1', 'f2']

        mfe = MFE(features=feature_names, groups=['complexity'])
        mfe.fit(reduced_features, sampled_labels)

        mfe_names, mfe_values = mfe.extract()
        complexity_features = dict(zip(mfe_names, mfe_values))
        
        metafeature_dict.update(quality_features)
        metafeature_dict.update(complexity_features)

    return metafeature_dict
    
def flatten_features(features, flatten : list = ['pixel_mean','pixel_std']):
    """
    Transforma as meta-características da lista `flatten` em meta-características individuais por canal, retornando um novo dicionário.
    """

    flat = {}

    for key in features.keys():
        if key not in flatten:
            flat[key] = features[key]

    for i, channel in enumerate(["r", "g", "b"]):
        flat[f"pixel_mean_{channel}"] = features["pixel_mean"][i]
        flat[f"pixel_std_{channel}"] = features["pixel_std"][i]

    return flat