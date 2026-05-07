


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














# utilities for creating mixed-modality contigs from real metagenomic sequences

##################
def save_as_omg_parquet(rows, output_path):
    """
    Saves a list of OMG-formatted dictionaries to a Parquet file.
    """
    # 1. Convert list of dicts to a Pandas DataFrame
    df = pd.DataFrame(rows)
    # print(df)
    # exit()
    
    # 2. Define the PyArrow Schema (Ensures list types are preserved correctly)
    # This prevents 'object' type issues which can break Hugging Face loaders
    schema = pa.schema([
        ('contig_id', pa.string()),
        ('IGS_seqs', pa.list_(pa.string())),
        ('CDS_seqs', pa.list_(pa.string())),
        ('CDS_ids', pa.list_(pa.string())),
        ('IGS_position_ids', pa.list_(pa.int64())),
        ('CDS_position_ids', pa.list_(pa.int64())),
        ('CDS_orientations', pa.list_(pa.bool_())),
        #('CDS_labels', pa.list_(pa.string())),
        # Add any metadata fields you've included
        #('CDS_ids', pa.list_(pa.string())), 
        #('IGS_ids', pa.list_(pa.string()))
    ])
    
    # 3. Convert DataFrame to Arrow Table
    table = pa.Table.from_pandas(df, schema=schema)
    
    # 4. Write to Parquet with Snappy compression (OMG default)
    pq.write_table(table, output_path, compression='snappy')
    print(f"Dataset saved successfully to {output_path}")



def load_multiple_fastas(file_paths):
    """
    Combines sequences from multiple FASTA files into one dictionary.
    
    Args:
        file_paths (list): List of paths to .fasta or .faa files.
        
    Returns:
        dict: Mapping of header IDs to sequence strings.
    """
    combined_dict = {}
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        # Parse each file and update the master dictionary
        # .id gets the unique identifier (everything before the first space)
        # str(record.seq) converts the Bio.Seq object to a standard string
        for record in SeqIO.parse(file_path, "fasta"):
            if record.id in combined_dict:
                print(f"Collision Warning: ID '{record.id}' already exists. Overwriting.")
            
            combined_dict[record.id] = str(record.seq)
            
    return combined_dict



def create_mag_dataset(mag_data_path, raw_mags_path, all_raw_mags_path, save_path):
    df = pd.read_csv(mag_data_path)
    df = df[['CSMAG_ID', 'Sequence_ID', 'contig_ID', 'Start', 'End', 'Strand', 'Sequence', 'Sequence_nucleotide', 'KO']]
    df.dropna(subset=['KO'], inplace=True)
    print(df)

    oris = {-1: False, 1: True}

    all_mag_fasta_files = os.listdir(all_raw_mags_path)
    mag_fastas = [all_raw_mags_path + file_name for file_name in all_mag_fasta_files]
    mag_id_seq_table = load_multiple_fastas(mag_fastas)
    all_contigs = []

    unique_contig_ids = df['contig_ID'].unique()
    for contig_id in unique_contig_ids:
        print(f"processing {contig_id}")
        contig_df = df[df['contig_ID'] == contig_id]
        raw_contig_seq_id = '_'.join(contig_df.iloc[0].Sequence_ID.split('_')[:-1])
        curr_contig = mag_id_seq_table[raw_contig_seq_id]
        cds_intervals = list(zip(contig_df.Start.astype(int), contig_df.End.astype(int)))
        row_data = format_to_omg_row(contig_id, curr_contig, cds_intervals)
        row_data['CDS_seqs'] = list(contig_df['Sequence'])
        row_data['CDS_ids'] = list(contig_df['Sequence_ID'])
        row_data['CDS_labels'] = list(contig_df['KO'])
        row_data['CDS_orientations'] = [oris[i] for i in contig_df['Strand']]
        all_contigs.append(row_data)

    save_as_omg_parquet(all_contigs, save_path)



def create_mag_dataset_cami(mag_data_path):
    df = pd.read_csv(mag_data_path) #, nrows=10000)
    df = df[['CSMAG_ID', 'Sequence_ID', 'contig_ID', 'Start', 'End', 'Strand', 'Sequence', 'Sequence_nucleotide']]
    #df.dropna(subset=['KO'], inplace=True)
    print(df)

    oris = {-1: False, 1: True}

    all_mag_fasta_files = os.listdir('data/') #contigs_filtered_postprodigal.fasta')
    mag_fastas = ['data/' + file_name for file_name in all_mag_fasta_files if file_name == 'contigs_filtered_postprodigal.fasta']
    #print(mag_fastas)
    #exit()
    mag_id_seq_table = load_multiple_fastas(mag_fastas)
    all_contigs = []

    unique_contig_ids = df['contig_ID'].unique()
    for contig_id in unique_contig_ids:
        print(f"processing {contig_id}")
        contig_df = df[df['contig_ID'] == contig_id]
        raw_contig_seq_id = '_'.join(contig_df.iloc[0].Sequence_ID.split('_')[:-1])
        curr_contig = mag_id_seq_table[raw_contig_seq_id]
        cds_intervals = list(zip(contig_df.Start.astype(int), contig_df.End.astype(int)))
        row_data = format_to_omg_row(contig_id, curr_contig, cds_intervals)
        row_data['CDS_seqs'] = list(contig_df['Sequence'])
        row_data['CDS_ids'] = list(contig_df['Sequence_ID'])
        #row_data['CDS_labels'] = list(contig_df['KO'])
        row_data['CDS_orientations'] = [oris[i] for i in contig_df['Strand']]
        all_contigs.append(row_data)
        

    save_as_omg_parquet(all_contigs, 'cami-mags.parquet')

#####################




