


import torch
from transformers import (
    AutoModel,
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    DefaultDataCollator,
    EsmModel,
    BatchEncoding,
)
from transformers.modeling_outputs import BaseModelOutput
from datasets import load_dataset
from evaluate import load
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from maglm.model import BiMambaModel
from tqdm import tqdm
import json
import os
import seaborn as sns
from typing import Dict, List, Literal, Optional
from torch import Tensor
from maglm.util import (
    print_model_size, 
    layer_breakdown,
    ProtOGDataset,
    OMGDataset,
    EsmProteinKODataset,
    ProteinKODataset,
    construct_glm_tokenizer,
)
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality
import dgeb
from torch.optim import Adafactor, AdamW
from maglm.config import (
    BiMambaSequenceClassifierConfig,
    BiMambaConfig,
    gLMConfig,
    ESMConfig,
)
from maglm.model import (
    BiMambaMLM,
    BiMambaForSequenceClassification,
    ESM2ForSequenceClassification,
    gLM2ForSequenceClassification,
)


import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    matthews_corrcoef, 
    top_k_accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import RidgeClassifier

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from dgeb.eval_utils import ForwardHook, pool
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams





# INITIALIZE GH200 MEMORY MANAGEMENT
# 1. Configure RMM for GH200 Unified Memory
# 'managed_memory=True' is the critical flag that allows the GPU to 
# transparently access the Grace CPU's huge RAM (480GB+).

initial_size = 54 * 1024**3

rmm.reinitialize(
    pool_allocator=True,
    managed_memory=True,  # <--- MUST ENABLE FOR GH200 OVERSUBSCRIPTION
    initial_pool_size=initial_size, # Let it grow dynamically
)

# 2. Hot-swap the PyTorch allocator
# From this point on, every .to('cuda') or torch.tensor(..., device='cuda')
# uses RMM instead of the default PyTorch caching allocator.
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

print(f"Current Allocator: {torch.cuda.memory.get_allocator_backend()}")




def get_best_metric(metrics_dict, task_name):
    """
    Selects the most statistically relevant metric for a DGEB task.
    Priority: top_corr > f1 > v_measure > accuracy > others
    """
    # these are the primary metrics for each task as specfied in the DGEB paper
    task_metrics = {
        "BacArch BiGene":                    'f1',
        "MIBiG Classification":              'f1',
        "Cyano Operonic Pair":               'cos_sim_ap',
        "MopB Clustering":                   'v_measure',
        "FeFeHydrogenase Phylogeny":         'top_corr',
        "EC Classification":                 'f1',
        "16S Bacterial Phylogeny":           'top_corr',
        "RpoB Archaeal Phylogeny":           'top_corr',
        "18S Eukaryotic Phylogeny":          'top_corr',
        "E.coli Operonic Pair":              'cos_sim_ap',
        "RpoB Bacterial Phylogeny":          'top_corr',
        "16S Archaeal Phylogeny":            'top_corr',
        "E.coli RNA Clustering":             'v_measure',
        "ModAC Paralogy BiGene":             'recall_at_50',
        "Vibrio Operonic Pair":              'cos_sim_ap',
        "Convergent Enzymes Classification": 'f1',
        "Arch Retrieval":                    'map_at_5',
        "Euk Retrieval":                     'map_at_5',
    }
    
    # Check if any priority metrics exist in the dictionary
    metric = task_metrics[task_name]
    if metric in metrics_dict:
        return metric
    else:
        print("MISSING METRICS")
        exit()
    # Fallback: if none of the above, just take the first available key
    #return list(metrics_dict.keys())[0]


def plot_all_checkpoints_comparison(full_data, filename):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    ckpt_display = {
        "checkpoint": "Nammu-C",
        "nucleotide":  "NTv2-250M",
    }
    def display_name(ckpt):
        for key, name in ckpt_display.items():
            if key in ckpt:
                return name
        return ckpt

    checkpoints = list(full_data.keys())
    task_names  = list(full_data[checkpoints[0]].keys())

    task_sort_val = {
        t: np.mean([list(full_data[c][t]["last"].values())[0] for c in checkpoints])
        for t in task_names
    }
    task_names = sorted(task_names, key=lambda x: task_sort_val[x], reverse=True)

    mid_data, last_data = {}, {}
    for ckpt in checkpoints:
        mid_data[ckpt], last_data[ckpt] = [], []
        for task in task_names:
            m_key = get_best_metric(full_data[ckpt][task]["mid"],  task)
            l_key = get_best_metric(full_data[ckpt][task]["last"], task)
            mid_data[ckpt].append(full_data[ckpt][task]["mid"][m_key])
            last_data[ckpt].append(full_data[ckpt][task]["last"][l_key])

    x         = np.arange(len(task_names))
    num_ckpts = len(checkpoints)
    total_group_width = 0.8
    bar_width         = total_group_width / (num_ckpts * 2)

    fig_height = max(2.6, 0.18 * len(task_names) + 1.6)
    fig, ax    = plt.subplots(figsize=(6.75, fig_height))

    colors = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161"]

    for i, ckpt in enumerate(checkpoints):
        name   = display_name(ckpt)
        color  = colors[i % len(colors)]
        offset = (i * 2 * bar_width) - (total_group_width / 2) + (bar_width / 2)

        ax.bar(x + offset, mid_data[ckpt], bar_width,
               label=f"{name} (Mid)", color=color,
               edgecolor="white", linewidth=0.3)

        ax.bar(x + offset + bar_width, last_data[ckpt], bar_width,
               label=f"{name} (Last)", facecolor=color, alpha=0.35,
               edgecolor=color, linewidth=0.4, hatch="///")

    ax.set_ylabel("Primary metric score")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha="right", fontsize=5)
    ax.tick_params(length=2.5)
    ax.set_xlim(-0.5, len(task_names) - 0.5)
    ax.set_title('DGEB Protein Tasks')

    ax.grid(axis="y", color="0.92", linewidth=0.4)
    ax.set_axisbelow(True)

    # Legend: one column per model, Mid row on top / Last row below
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               ncol=num_ckpts,
               bbox_to_anchor=(0.5, -0.02),
               frameon=False,
               handlelength=1.6,
               columnspacing=1.2,
               handletextpad=0.4)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.28)

    base = filename.rsplit(".", 1)[0]
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_benchmarks(path_to_results, figname):

    all_metrics_dict = {}
    for model in path_to_results:
        model_name = model.split('/')[-2]
        all_metrics_dict[model_name] = {}

    for model in path_to_results:
        all_results = os.listdir(f"{model}")
        print(model)
        model_name = model.split('/')[-2]
        print(model_name)
        for f in all_results:

            with open(model + '/' + f, 'r') as f:
                res = json.load(f)

            num_layers = res['model']['num_layers'] - 1
            task_name = res['task']['display_name']
            all_metrics_dict[model_name][task_name] = {'mid': {}, 'last': {}}
            #print(task_name)
            for result in res['results']:
                layer = result['layer_number']
                for metric in result['metrics']:
                    if layer < num_layers:
                        all_metrics_dict[model_name][task_name]['mid'][metric['id']] = metric['value']
                    else:
                        all_metrics_dict[model_name][task_name]['last'][metric['id']] = metric['value']


    plot_all_checkpoints_comparison(all_metrics_dict, figname)


def plot_cami_results_lines(
    ge4096_knn_csv,  ge4096_ridge_csv,
    all_knn_csv,     all_ridge_csv,
    figname,
):
    
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # --- Data --------------------------------------------------------------------
    def load_df(knn_csv, ridge_csv):
        df_knn              = pd.read_csv(knn_csv)
        df_ridge            = pd.read_csv(ridge_csv)
        df_knn["classifier"]   = "kNN"
        df_ridge["classifier"] = "Ridge"
        df = pd.concat([df_knn, df_ridge], ignore_index=True)
        df["label"] = pd.Categorical(df["label"], categories=TAX_ORDER, ordered=True)
        return df.sort_values(["classifier", "model", "label"])

    TAX_ORDER  = ["class", "order", "family", "genus", "species"]
    TAX_LABELS = ["Class", "Order", "Family", "Genus", "Species"]
    metrics    = ["top1_acc", "f1_macro", "mcc", "precision", "recall"]
    metric_labels = {
        "top1_acc":  "Top-1 Accuracy",
        "f1_macro":  "Macro F1",
        "mcc":       "MCC",
        "precision": "Precision",
        "recall":    "Recall",
    }
    classifiers = ["kNN", "Ridge"]

    df_ge4096 = load_df(ge4096_knn_csv, ge4096_ridge_csv)
    df_all    = load_df(all_knn_csv,    all_ridge_csv)

    model_styles = {
        "embedding_nammu": dict(color="#0173B2", marker="o", label="Nammu"),
        "embedding_glm":   dict(color="#DE8F05", marker="s", label="gLM2"),
    }

    global_ymax = max(df_ge4096[metrics].max().max(), df_all[metrics].max().max())
    y_lim = (0, global_ymax * 1.08)

    datasets = [
        (df_ge4096, "solid"),   # ≥4096 bp  — solid lines
        (df_all,    "dashed"),  # all lengths — dashed lines
    ]

    # --- Figure: 2 rows × 5 cols -------------------------------------------------
    fig, axes = plt.subplots(
        len(classifiers), len(metrics),
        figsize=(7.0, 3.0),
        sharex=True, sharey=True,
    )

    x = np.arange(len(TAX_ORDER))

    for row, classifier in enumerate(classifiers):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            for df, linestyle in datasets:
                sub = df[df["classifier"] == classifier]
                for model, group in sub.groupby("model"):
                    s = model_styles[model]
                    ax.plot(x, group[metric].values,
                            color=s["color"], marker=s["marker"],
                            linestyle=linestyle)

            if row == 0:
                ax.set_title(metric_labels[metric])

            ax.set_ylim(*y_lim)
            ax.tick_params(length=2.5)
            ax.grid(axis="y", color="0.92", linewidth=0.4)
            ax.set_axisbelow(True)

            if row == len(classifiers) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(TAX_LABELS, rotation=35, ha="right")

        axes[row, 0].set_ylabel(f"{classifier} Score")

    # --- Legend ------------------------------------------------------------------
    # Row 1: model colors  |  Row 2: line styles for datasets
    legend_handles = [
        Line2D([0], [0], color=s["color"], marker=s["marker"],
               linestyle="solid", label=s["label"])
        for s in model_styles.values()
    ] + [
        Line2D([0], [0], color="0.4", linestyle="solid",  label="≥4096 bp"),
        Line2D([0], [0], color="0.4", linestyle="dashed", label="All lengths"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.20,
                        wspace=0.20, hspace=0.22)

    base = figname.rsplit(".", 1)[0]
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def ko_bench():
    contig_filepath = None
    protein_filepath = None

    df_contig = pd.read_parquet(contig_filepath)
    df_protein = pd.read_parquet(protein_filepath)
    #df_combined = df_protein
    #print(df_contig)
    #print(df_protein)

    colname_contig = '_'.join(contig_filepath.split('/')[-1].split('_')[:5])
    colname_prot = '_'.join(protein_filepath.split('/')[-1].split('_')[:5])

    df_combined                      = df_contig.merge(df_protein, left_on='ids', right_on='ids')
    df_combined                      = df_combined[['ids', 'label_x', 'embedding_x', 'embedding_y']]
    df_combined.columns              = ['ids', 'label', colname_contig, colname_prot]
   

    train_ko_models(df_combined, embedding_col=colname_prot)
    train_ko_models(df_combined, embedding_col=colname_contig)




def train_ko_models(df, embedding_col, label_col, model_type='knn', test_df=None):

    print(f"input: {embedding_col}")
    # Filter for labels with enough samples to allow for a 80/20 stratified split
    df_filtered = df[df.groupby(label_col)[label_col].transform('count') > 10].copy()

    train_df, test_df = train_test_split(
        df_filtered, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_filtered[label_col]
    )

    # print(train_df.head())
    # print(test_df.head())

    #print(len(train_df[label_col].unique()))
    #exit()

    X_train = np.stack(train_df[embedding_col].values)
    y_train = train_df[label_col].values
    X_test = np.stack(test_df[embedding_col].values)
    y_test = test_df[label_col].values

    if model_type == 'ridge':
        clf = RidgeClassifier(
            alpha=1.0,
            class_weight='balance',
        )
    
    elif model_type == 'knn':
        clf = KNeighborsClassifier(
                n_neighbors=1, 
                metric='cosine', 
                n_jobs=-1,
                weights='distance',
                #algorithm='ball_tree',
                #p=2,
        )


    model = make_pipeline(clf)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    # --- CALCULATE METRICS ---
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    
    # Top-K scores (ensure labels match the classifier's internal class order)
    # top5_acc = top_k_accuracy_score(y_test, y_proba, k=5, labels=knn.classes_)
    # top10_acc = top_k_accuracy_score(y_test, y_proba, k=10, labels=knn.classes_)

    print(f"\n{'='*40}")
    print(f"kNN Results (n_test={len(y_test)})")
    print(f"{'='*40}")
    print(f"Accuracy (Top-1): {acc:.4f}")
    # print(f"Top-5 Accuracy:   {top5_acc:.4f}")
    # print(f"Top-10 Accuracy:  {top10_acc:.4f}")
    print(f"Macro F1 Score:   {f1:.4f}")
    print(f"MCC:              {mcc:.4f}")
    print(f"Precision:        {prec:.4f}")
    print(f"Recall:          {rec:.4f}")

    print(f"{'='*40}\n")

    metrics = {
        'top1_acc': float(acc),
        # 'top5_acc': float(top5_acc),
        # 'top10_acc': float(top10_acc),
        'f1_macro': float(f1),
        'mcc': float(mcc),
    }





# To run KO tasks, needs parquet files of embeddings from models. 
# process described in README
# ko_bench()


# To plot CAMI line plots
plot_cami_results_lines(
    ge4096_knn_csv="results/all_metrics_ge4096_knn.csv",
    ge4096_ridge_csv="results/all_metrics_ge4096_ridge.csv",
    all_knn_csv="results/all_metrics_full_contig_knn.csv",
    all_ridge_csv="results/all_metrics_full_contig_ridge.csv",
    figname="results/cami_lines_combined.png",
)

# To plot DGEB DNA Tasks
plot_benchmarks([
    'results/dgeb_nammu_omg_dna/checkpoint-250000/',
    'results/dgeb_nammu_omg_dna/gLM2-150M/',
    'results/dgeb_nammu_omg_dna/caduceus-ps/',
    'results/dgeb_nammu_omg_dna/nucleotide-transformer-v2-250m-multi-species/',
    ],
     figname='results/dgeb_dna_final.png',
)

# To plot DGEB Amino Tasks
plot_benchmarks([
    'results/dgeb_nammu_omg/checkpoint-250000/',
    'results/dgeb_glm2_omg/gLM2-150M/',
    'results/dgeb_nammu_omg/esm2_t30_150M_UR50D/',
    ],
    figname='results/dgeb_protein_final.png',
)

