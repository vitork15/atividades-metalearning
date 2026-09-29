from utils import *
import pandas as pd
import os
import csv
from scipy.stats import rankdata, wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from tqdm import tqdm
import warnings
from huggingface_hub import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scikit_posthocs as sp

def main():
    
    matplotlib.use("Agg")
    
    # remove warnings desnecessários
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="torch"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*looks like a TorchScript archive.*"
    )
    logging.set_verbosity_error()
    
    disable_tqdm = False
    
    # extração de metafeatures + resultados em csv
    dataset_csv = 'dataset_metafeatures_expanded.csv'      
    datasets = get_datasets()

    if os.path.exists(dataset_csv):
        dataset_df = pd.read_csv(dataset_csv)
    else:
        dataset_df = pd.DataFrame()

    if not dataset_df.empty:
        datasets_presentes = set(dataset_df["dataset"])
    else:
        datasets_presentes = set()

    for dataset_name, dataset_loader in tqdm(datasets,desc="Gerando metafeatures",disable=disable_tqdm):
        if dataset_name in datasets_presentes:
            continue

        transform = None

        if dataset_name in ['Caltech-101', 'Caltech-256']:
            transform = transforms.Lambda(lambda img: img if img.mode == "RGB" else img.convert("RGB"))
        elif dataset_name == 'FashionMNIST':
            transform = transforms.Grayscale(num_output_channels=3)

        dataset = dataset_loader(transform=transform)
        metafeatures = get_dataset_metafeatures(dataset, extended=True, subsample=True)
        flat = flatten_features(metafeatures)
        flat['dataset'] = dataset_name

        row_df = pd.DataFrame([flat])

        row_df.to_csv(
            dataset_csv,
            mode='a',
            header=not os.path.exists(dataset_csv),
            index=False
        )

        datasets_presentes.add(dataset_name)
    
    metafeatures_df = pd.read_csv('dataset_metafeatures.csv')
    metafeatures_df = metafeatures_df[metafeatures_df["dataset"] != "Oxford-IIIT Pet"] # removendo oxford pet temporariamente
    ext_metafeatures_df = pd.read_csv('dataset_metafeatures_expanded.csv').drop(columns=['l1.sd','l2.sd','l3.sd','f1.sd','f2.sd'])
    
    # csv de resultados
    fold_df = pd.read_csv('metadata.csv')
    
    metrics = [
        "acc_top1",
        "acc_top5",
        "f1",
        "train_time",
        "val_time"
    ]
    
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
    
    corr = ext_metafeatures_df.select_dtypes(include="number").corr(method="pearson")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=300)
    
    max_depth = 4
    
    regr_alexnet1l = RandomForestRegressor(max_depth=max_depth, random_state=1234)
    regr_alexnetnf = RandomForestRegressor(max_depth=max_depth, random_state=1234)
    regr_resnet1l = RandomForestRegressor(max_depth=max_depth, random_state=1234)
    regr_resnetnf = RandomForestRegressor(max_depth=max_depth, random_state=1234)
    regr_inception1l = RandomForestRegressor(max_depth=max_depth, random_state=1234)
    
    models = [('AlexNet-1L',regr_alexnet1l),('ResNet18-1L',regr_resnet1l),('Inception-1L',regr_inception1l),('AlexNet-NF',regr_alexnetnf),('ResNet18-NF',regr_resnetnf)]
    
    top_n = 5
    
    # plot de importancia de feature (todos os modelos)
    features = ext_metafeatures_df.drop(columns=["dataset"])

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 2)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),  
    ]

    for ax, (model_name, model) in zip(axes, models):

        probs = prob_matrix.loc[
            ext_metafeatures_df["dataset"],
            model_name
        ].values

        model.fit(features, probs)

        importances = (
            pd.Series(
                model.feature_importances_,
                index=features.columns
            )
            .sort_values(ascending=False)
            .head(top_n)
            .sort_values()
        )

        importances.plot(
            kind="barh",
            ax=ax
        )

        ax.set_xlabel("Importância (MDI)")
        ax.set_ylabel("Feature")
        ax.set_title(model_name)

    fig.suptitle(
        f"Top {top_n} Features por Modelo",
        fontsize=16
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(
        "importances_all_models.png",
        dpi=500,
        bbox_inches="tight"
    )

    ext_acc_list = []
    spearman = []
    loss_curves = []
    loss_area = []
    
    loo = LeaveOneOut()
    for i, (train_id, test_id) in enumerate(loo.split(ext_metafeatures_df)):
        
        train_features = ext_metafeatures_df.iloc[train_id].drop(columns=['dataset'])
        test_features = ext_metafeatures_df.iloc[test_id].drop(columns=['dataset'])
            
        train_datasets = ext_metafeatures_df.iloc[train_id]['dataset']        
        pred = {}
        
        for model_name, model in models:
        
            train_acc = prob_matrix.loc[
                train_datasets,
                model_name
            ].values
            
            model.fit(train_features, train_acc)
            pred[model_name] = model.predict(test_features)[0]

        true_acc = prob_matrix.iloc[test_id[0]]
        true_rank = rank_matrix.iloc[test_id[0]]
        
        pred_acc = pd.Series(pred, index=true_rank.index)
        pred_rank = pred_acc.rank(ascending=False, method="min")
        ext_acc_list.append(pred_acc)
        
        # spearman
        rho, _ = spearmanr(true_rank, pred_rank)
        spearman.append(rho)
        
        # curva de loss
        best = true_acc.max()
        
        ranked_models = pred_rank.sort_values().index
        ranked_by_model = true_acc[ranked_models]
        
        best_until_t = ranked_by_model.cummax()

        # loss(t) = max_j P_ij - max_{a <= t} P_ia
        loss = best - best_until_t
        loss_curves.append(loss)
        
        # area da curva
        loss_area.append(loss.sum())
    
    print("Valor médio do coeficiente de Spearman (matriz extendida) - regressão na acurácia:", np.mean(spearman))
    print("Valor médio da área da curva de loss (matriz extendida) - regressão na acurácia:", np.mean(loss_area))
    print("---------------------")
    
    orig_acc_list = []
    spearman = []
    loss_curves = []
    loss_area = []
    
    for i, (train_id, test_id) in enumerate(loo.split(metafeatures_df)):
    
        train_features = metafeatures_df.iloc[train_id].drop(columns=['dataset'])
        test_features = metafeatures_df.iloc[test_id].drop(columns=['dataset'])
            
        train_datasets = metafeatures_df.iloc[train_id]['dataset']        
        pred = {}
        
        for model_name, model in models:
        
            train_acc = prob_matrix.loc[
                train_datasets,
                model_name
            ].values
            
            model.fit(train_features, train_acc)
            pred[model_name] = model.predict(test_features)

        true_acc = prob_matrix.iloc[test_id[0]]
        true_rank = rank_matrix.iloc[test_id[0]]
        
        pred_acc = pd.Series(pred, index=true_rank.index)
        pred_rank = pred_acc.rank(ascending=False, method="min")
        orig_acc_list.append(pred_acc)
        
        # spearman
        rho, _ = spearmanr(true_rank, pred_rank)
        spearman.append(rho)
        
        # curva de loss
        best = true_acc.max()
        
        ranked_models = pred_rank.sort_values().index
        ranked_by_model = true_acc[ranked_models]
        
        best_until_t = ranked_by_model.cummax()

        # loss(t) = max_j P_ij - max_{a <= t} P_ia
        loss = best - best_until_t
        loss_curves.append(loss)
        
        # area da curva
        loss_area.append(loss.sum())
    
    print("Valor médio do coeficiente de Spearman (matriz original) - regressão na acurácia:", np.mean(spearman))
    print("Valor médio da área da curva de loss (matriz original) - regressão na acurácia:", np.mean(loss_area))
    print("---------------------")
    
    spearman = []
    loss_curves = []
    loss_area = []
    
    loo = LeaveOneOut()
    for i, (train_id, test_id) in enumerate(loo.split(ext_metafeatures_df)):
        
        train_features = ext_metafeatures_df.iloc[train_id].drop(columns=['dataset'])
        test_features = ext_metafeatures_df.iloc[test_id].drop(columns=['dataset'])
            
        train_datasets = ext_metafeatures_df.iloc[train_id]['dataset']        
        pred = {}
        
        for model_name, model in models:
        
            train_ranking = rank_matrix.loc[
                train_datasets,
                model_name
            ].values
            
            model.fit(train_features, train_ranking)
            pred[model_name] = model.predict(test_features)[0]

        true_acc = prob_matrix.iloc[test_id[0]]
        true_rank = rank_matrix.iloc[test_id[0]]
        
        pred_rank = pd.Series(pred, index=true_rank.index)
        
        # spearman
        rho, _ = spearmanr(true_rank, pred_rank)
        spearman.append(rho)
        
        # curva de loss
        best = true_acc.max()
        
        ranked_models = pred_rank.sort_values().index
        ranked_by_model = true_acc[ranked_models]
        
        best_until_t = ranked_by_model.cummax()

        # loss(t) = max_j P_ij - max_{a <= t} P_ia
        loss = best - best_until_t
        loss_curves.append(loss)
        
        # area da curva
        loss_area.append(loss.sum())
    
    print("Valor médio do coeficiente de Spearman (matriz extendida) - regressão no ranking:", np.mean(spearman))
    print("Valor médio da área da curva de loss (matriz extendida) - regressão no ranking:", np.mean(loss_area))
    print("---------------------")
    
    spearman = []
    loss_curves = []
    loss_area = []
    
    loo = LeaveOneOut()
    for i, (train_id, test_id) in enumerate(loo.split(metafeatures_df)):
        
        train_features = metafeatures_df.iloc[train_id].drop(columns=['dataset'])
        test_features = metafeatures_df.iloc[test_id].drop(columns=['dataset'])
            
        train_datasets = metafeatures_df.iloc[train_id]['dataset']        
        pred = {}
        
        for model_name, model in models:
        
            train_ranking = rank_matrix.loc[
                train_datasets,
                model_name
            ].values
            
            model.fit(train_features, train_ranking)
            pred[model_name] = model.predict(test_features)[0]

        true_acc = prob_matrix.iloc[test_id[0]]
        true_rank = rank_matrix.iloc[test_id[0]]
        
        pred_rank = pd.Series(pred, index=true_rank.index)
        
        # spearman
        rho, _ = spearmanr(true_rank, pred_rank)
        spearman.append(rho)
        
        # curva de loss
        best = true_acc.max()
        
        ranked_models = pred_rank.sort_values().index
        ranked_by_model = true_acc[ranked_models]
        
        best_until_t = ranked_by_model.cummax()

        # loss(t) = max_j P_ij - max_{a <= t} P_ia
        loss = best - best_until_t
        loss_curves.append(loss)
        
        # area da curva
        loss_area.append(loss.sum())
    
    print("Valor médio do coeficiente de Spearman (matriz original) - regressão no ranking:", np.mean(spearman))
    print("Valor médio da área da curva de loss (matriz original) - regressão no ranking:", np.mean(loss_area))
        
    # diagrama de diff critica
    
    extended_pred_acc = pd.DataFrame(ext_acc_list)
    original_pred_acc = pd.DataFrame(orig_acc_list)
    
    extended_predicted_ranks = extended_pred_acc.rank(
        axis=1,
        ascending=False,
        method="average"
    )
    original_predicted_ranks = extended_pred_acc.rank(
        axis=1,
        ascending=False,
        method="average"
    )
    
    extended_avg_rank = extended_predicted_ranks.mean(axis=0)
    original_avg_rank = original_predicted_ranks.mean(axis=0)
    
    extended_p_values = sp.posthoc_nemenyi_friedman(extended_pred_acc)

    fig, ax = plt.subplots(figsize=(10, 4))

    sp.critical_difference_diagram(
        extended_avg_rank,
        extended_p_values,
        ax=ax
    )

    fig.tight_layout()
    fig.savefig(
        "ext_critical_difference_diagram.pdf",
        bbox_inches="tight"
    )
    
    original_p_values = sp.posthoc_nemenyi_friedman(original_pred_acc)

    fig, ax = plt.subplots(figsize=(10, 4))

    sp.critical_difference_diagram(
        original_avg_rank,
        original_p_values,
        ax=ax
    )

    fig.tight_layout()
    fig.savefig(
        "orig_critical_difference_diagram.pdf",
        bbox_inches="tight"
    )
    
if __name__ == "__main__":
    main()