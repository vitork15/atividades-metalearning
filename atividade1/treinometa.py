from utils import *
import pandas as pd
import os
import csv
from tqdm import tqdm

def main():
    
    disable_tqdm = False
    
    dataset_csv = 'dataset_metafeatures.csv'      
    datasets = get_datasets()

    if os.path.exists(dataset_csv):
        dataset_df = pd.read_csv(dataset_csv)
    else:
        dataset_df = pd.DataFrame()
        
    if not dataset_df.empty:
        datasets_presentes = set(dataset_df["dataset"])
    else:
        datasets_presentes = set()

    new_entries = []

    for dataset_name, dataset_loader in tqdm(datasets, desc="Gerando metafeatures", disable=disable_tqdm):
        if dataset_name in datasets_presentes:
            continue

        transform=None
        if dataset_name in ['Caltech-101','Caltech-256']:
            transform = transforms.Lambda(lambda img: img if img.mode == "RGB" else img.convert("RGB"))
        elif dataset_name in ['FashionMNIST']:
            transform = transforms.Grayscale(num_output_channels=3)
        
        dataset = dataset_loader(transform=transform)
        metafeatures = get_dataset_metafeatures(dataset)   
        flat = flatten_features(metafeatures)
        flat['dataset'] = dataset_name
        
        new_entries.append(flat)

    if new_entries:
        new_dataset_df = pd.DataFrame(new_entries)
        dataset_df = pd.concat([dataset_df, new_dataset_df], ignore_index=True)

    dataset_df.to_csv(dataset_csv, index=False)
    training_df = pd.read_csv('metadata.csv')

if __name__ == "__main__":
    main()