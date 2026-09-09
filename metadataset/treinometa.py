from utils import *
import pandas as pd
import os
import csv
from scipy.stats import rankdata, wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests
from itertools import combinations
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
    
    fold_df = pd.read_csv('metadata.csv')
    
    metrics = [
        "acc_top1",
        "acc_top5",
        "f1",
        "train_time",
        "val_time"
    ]

    aggregated_df = ( # matriz P representada por um lookup (i,j)
        fold_df
        .groupby(["dataset", "model"])[metrics]
        .agg({
            "acc_top1": lambda x: f"{100*x.mean():.2f}% ± {100*x.std():.2f}%",
            "acc_top5": lambda x: f"{100*x.mean():.2f}% ± {100*x.std():.2f}%",
            "f1": lambda x: f"{100*x.mean():.2f}% ± {100*x.std():.2f}%",
            "train_time": lambda x: f"{x.mean():.2f}s ± {x.std():.2f}s",
            "val_time": lambda x: f"{x.mean():.2f}s ± {x.std():.2f}s",
        })
        .reset_index()
    )
    
    prob_df = (
        fold_df
        .groupby(["dataset", "model"])
        .agg(
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in metrics
            },
            **{
                f"{metric}_std": (metric, "std")
                for metric in metrics
            }
        )
        .reset_index()
    )
    
    results_df = prob_df.copy()
    
    results_df["rank"] = (
        results_df
        .groupby("dataset")["acc_top1_mean"]
        .transform(lambda x: rankdata(-x, method="average"))
    )
    
    rank_df = results_df[["dataset", "model", "rank"]].copy() # matriz R representada por um lookup (i,j)
    
    #print(aggregated_df.drop(columns=["acc_top5","f1","train_time","val_time"]).to_latex())
    #print(rank_df.to_latex())
    
    rank_matrix = results_df.pivot(
        index="dataset",
        columns="model",
        values="rank"
    )
    
    prob_matrix = prob_df.pivot(
        index="dataset",
        columns="model",
        values="acc_top1_mean"
    )
    
    average_rank = rank_matrix.mean(axis=0)
    median_rank = rank_matrix.median(axis=0)
    
    #print(average_rank)
    #print(median_rank)

    models = fold_df["model"].unique()
    
    significance_level = 0.1

    wins = []

    for dataset in fold_df["dataset"].unique():

        dataset_df = fold_df[fold_df["dataset"] == dataset]

        comparisons = []

        for model_a, model_b in combinations(models, 2):

            a = dataset_df[dataset_df["model"] == model_a]["acc_top1"].values

            b = dataset_df[dataset_df["model"] == model_b]["acc_top1"].values

            stat, p = wilcoxon(a, b)

            comparisons.append({
                "model_a": model_a,
                "model_b": model_b,
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "p_value": p
            })

        p_values = [x["p_value"] for x in comparisons]

        for comparison, p_value in zip(comparisons, p_values):

            comparison["significant"] = p_value < significance_level

            if comparison["significant"]:
                if comparison["mean_a"] > comparison["mean_b"]:
                    comparison["winner"] = comparison["model_a"]
                else:
                    comparison["winner"] = comparison["model_b"]
            else:
                comparison["winner"] = None

            comparison["dataset"] = dataset

            wins.append(comparison)
            
    wins_df = pd.DataFrame(wins)
            
    significant_wins = (
        wins_df[wins_df["significant"]]
        .groupby(["dataset", "winner"])
        .size()
        .unstack(fill_value=0)
    )
    
    total_wins = significant_wins.sum(axis=0)

    significant_rank = pd.Series(
        rankdata(-total_wins.values, method="average"),
        index=total_wins.index,
        name="rank"
    )
        
    #print(significant_rank)
    
    # calcular o spearman de ranks fixos é o equivalente a calcular a media do spearman do rank real com o rank fixo
    avg_spear = spearmanr(average_rank, rank_matrix, axis=1).statistic.mean()
    median_spear = spearmanr(median_rank, rank_matrix, axis=1).statistic.mean()
    signif_spear = spearmanr(significant_rank, rank_matrix, axis=1).statistic.mean()
        
    print(avg_spear, median_spear, signif_spear)
    
    
if __name__ == "__main__":
    main()