from modelos import *
from sklearn.model_selection import StratifiedKFold
from torch import nn, cuda
import torch
import numpy as np
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, STL10, FashionMNIST, DTD, Places365, SUN397, Caltech101, Caltech256, EuroSAT, GTSRB, Food101, StanfordCars, FGVCAircraft, OxfordIIITPet, Flowers102
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from tqdm import tqdm
from functools import partial
import time
import os
import pandas as pd
import csv
from utils import *
import matplotlib.pyplot as plt

def main():
    
    csv_file = 'metadata.csv'
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, "a", newline="") as f: # header csv
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "dataset",
                "model",
                "fold",
                "acc_top1",
                "acc_top5",
                "f1",
                "train_time",
                "val_time"
            ])
    
    if os.path.exists(csv_file):
        existing = pd.read_csv(csv_file)

        completed = set(
            zip(existing["dataset"], existing["model"])
        )
    else:
        completed = set()
    
    disable_tqdm=False
    
    standard_transform = ResNet18_Weights.IMAGENET1K_V1.transforms() # a transformação usada por todos os modelos pre-treinados é identica a da resnet18
    inception_transform = Inception_V3_Weights.IMAGENET1K_V1.transforms() # exceto pelo inception
    
    device = "cuda" if cuda.is_available() else "cpu"
    
    num_splits = 5
    batch_size = 64
    
    kfold = StratifiedKFold(
        n_splits=num_splits,
        shuffle=True
    )
    
    models = get_models() # definidos em utils
    datasets = get_datasets()
    
    for dataset_name, dataset_loader in datasets:
        for model_name, architecture in models:
            
            if (dataset_name, model_name) in completed:
                print(f"Pulando {dataset_name} + {model_name} - já foi executado")
                continue
            
            if model_name in ['Inception-1L','Inception-NF']:
                transform = inception_transform
            else:
                transform = standard_transform
            
            if dataset_name in ['Caltech-101','Caltech-256']:
                transform = transforms.Compose([
                transforms.Lambda(lambda img: img if img.mode == "RGB" else img.convert("RGB")),
                transform
                ])
            elif dataset_name in ['FashionMNIST']:
                transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transform
                ])
                
            dataset = dataset_loader(transform=transform)
            
            num_classes = get_num_classes(dataset)
            targets = get_labels(dataset)
            
            fold_metadata = []
            
            print(f"CV {num_splits}-fold - Dataset: {dataset_name} - Modelo: {model_name}")
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, targets)):

                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)

                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0
                )

                val_loader = DataLoader(
                    val_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )
                
                model = architecture(num_classes=num_classes)
                model.to(device)
                
                #nao funciona no windows
                #model.compile()
                
                parameters = model.trainable_parameters()
                optim = AdamW(params=parameters, weight_decay=1e-4, lr=1e-4)
                
                loss_func = nn.CrossEntropyLoss()
                
                if cuda.is_available():
                    cuda.synchronize()
                start = time.perf_counter()
                
                model.train()
                for images, labels in tqdm(train_loader, desc = f"Treino - Fold {fold+1}", disable=disable_tqdm):
                    
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optim.zero_grad()
                    
                    pred = model(images)
                    
                    loss = loss_func(pred, labels)
                    
                    loss.backward()
                    
                    optim.step()
                    
                if cuda.is_available():
                    cuda.synchronize()
                train_time = time.perf_counter()-start
                
                all_labels = []
                all_preds = []
                
                if cuda.is_available():
                    cuda.synchronize()
                start = time.perf_counter()
                
                model.eval()
                with torch.no_grad():
                    for images, labels in tqdm(val_loader, desc = f"Validação - Fold {fold+1}", disable=disable_tqdm):
                        
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        pred = model(images)
                        
                        all_preds.append(pred.cpu())
                        all_labels.append(labels.cpu())
                        
                    all_preds = torch.cat(all_preds).numpy()
                    all_labels = torch.cat(all_labels).numpy()
                    
                if cuda.is_available():
                    cuda.synchronize()
                val_time = time.perf_counter()-start

                hard_preds = all_preds.argmax(axis=1)
                acc_top1, f1 = accuracy_score(all_labels, hard_preds), f1_score(all_labels, hard_preds, average='macro')
                acc_top5 = top_k_accuracy_score(all_labels, all_preds, k=5, labels=np.arange(num_classes))
                
                fold_metadata.append([acc_top1, acc_top5, f1, train_time, val_time])
                
                #print(f"Acurácia Top1 - Fold {fold+1}:", acc_top1)
                #print(f"Acurácia Top5 - Fold {fold+1}:", acc_top5)
                #print(f"F1-score - Fold {fold+1}:", f1)
                #print(f"Tempo de treino - Fold {fold+1}:", train_time)
                #print(f"Tempo de validação - Fold {fold+1}:", val_time)
                
            with open("metadata.csv", "a", newline="") as file:
                writer = csv.writer(file)

                for fold, (acc_top1, acc_top5, f1, train_time, val_time) in enumerate(fold_metadata):
                    writer.writerow([
                        dataset_name,
                        model_name,
                        fold,
                        acc_top1,
                        acc_top5,
                        f1,
                        train_time,
                        val_time
                    ])

                file.flush()
            
                

if __name__ == "__main__":
    main()