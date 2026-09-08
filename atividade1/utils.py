import numpy as np
import torch
from functools import partial
from scipy.stats import kurtosis, skew
from sklearn.metrics import mutual_info_score
from modelos import *
from torchvision import transforms

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
                ('Oxford-IIIT Pet',partial(OxfordIIITPet,root="./datasets",split='trainval',download=True)),
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

def get_dataset_metafeatures(dataset):
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

    return {
        "num_samples": n_samples,
        "num_classes": n_classes,

        "area": height*width,

        "pixel_mean": mean.tolist(),
        "pixel_std": std.tolist(),

        "class_entropy": float(class_entropy),
    }
    
def flatten_features(features):
    """
    Calcula as meta-características de um dataset e retorna um dicionário `dict` com as meta-características.
    """

    flat = {}

    flat["num_samples"] = features["num_samples"]
    flat["num_classes"] = features["num_classes"]
    flat["area"] = features["area"]

    for i, channel in enumerate(["r", "g", "b"]):
        flat[f"pixel_mean_{channel}"] = features["pixel_mean"][i]
        flat[f"pixel_std_{channel}"] = features["pixel_std"][i]

    flat["class_entropy"] = features["class_entropy"]

    return flat