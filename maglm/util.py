import torch
import pandas as pd
import os
from Bio import SeqIO
import random
from tqdm import tqdm
import datetime
import logging
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib.pyplot as plt
from maglm.tokenizers import GenomicTokenizer
from prettytable import PrettyTable
import datasets
from torch.utils.data import Dataset, Subset, DataLoader

from transformers import (
    AutoModel, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
    EsmForSequenceClassification,
    EsmTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    AutoModelForMaskedLM,
    DefaultDataCollator,
)





LOGGER = logging.getLogger(__name__)

logging.basicConfig(
        format=f"[%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across multiple libraries."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

    # For deterministic behavior with CUDA backend (at the cost of some speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_size(model):
    total_params = model.num_parameters()
    trainable_params = model.num_parameters(only_trainable=True)
    
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"Trainable Ratio: {trainable_params/total_params:.2%}")


def layer_breakdown(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue # Skip frozen layers
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        
    print(table)
    print(f"Total Trainable Params: {total_params}")


class ProteinTokenizer:
    def __init__(self, include_dna=False):
        
        # Add amino acids first
        amino_acids = "ARNDCQEGHILKMFPSTWYV"
        self.aminos = "ARNDCQEGHILKMFPSTWYV"
        self.nucl = 'acgt'

        if include_dna: 
            amino_acids += "acgt"
            LOGGER.info('\nIncluding DNA tokens...')
        self.AA_to_idx = {}
        for i, aa in enumerate(amino_acids):
            self.AA_to_idx[aa] = i
        
        #print(self.AA_to_idx)
        # Add special tokens
        # Need to use <EOS> and <BOS> for start and end of CDS and use <BOC> and <EOC>
        # full contigs
        special_tokens = ["<PAD>", "<UNK>", "<MASK>"]

        for i, token in enumerate(special_tokens):
            self.AA_to_idx[token] = len(amino_acids) + i
        
        # Create reverse mapping
        self.idx_to_AA = {v: k for k, v in self.AA_to_idx.items()}
        self.vocab_size = len(self.idx_to_AA)

        #print(self.AA_to_idx)


    def encode(self, seq, mask=False):
        if seq is None or len(seq) == 0:
            raise ValueError
        
        if any(char.isdigit() for char in seq):
            raise ValueError("Sequence contains numbers")

        try:
            if mask:
                tokenized = [self.AA_to_idx.get(aa, self.AA_to_idx['<UNK>']) 
                            for aa in seq]
            else:
                tokenized = [self.AA_to_idx.get("<bos>")] + [self.AA_to_idx.get(aa, self.AA_to_idx['<UNK>']) for aa in seq] + [self.AA_to_idx.get("<eos>")]
                seq_tags = [None for aa in seq] #[self.AA_to_idx['<CDS>'] if aa in self.aminos else self.AA_to_idx['<IGS>'] for aa in seq]

        except KeyError:
            LOGGER.info(seq)
            raise KeyError
        return torch.tensor(tokenized) #, torch.tensor(seq_tags)

    def decode(self, int_seq):
        return ''.join(self.idx_to_AA[idx] for idx in int_seq)


class MAGDatasetIGS(torch.utils.data.Dataset):
    def __init__(self, fasta_data_dir, filename, is_mask=False, mask_rate=0.15, n=-1, step_per_len_increase=500, max_seq_len=-1, include_dna=True, get_headers=False):

        self.tok = ProteinTokenizer(include_dna=include_dna)
        self.is_mask = is_mask

        self.mask = is_mask
        self.mask_rate = mask_rate
        self.sequences = []
        self.masked_seqs = []
        self.seq_pos_ids = []
        self.headers = []
        self.maxlen = 0
        self.max_seq_len = max_seq_len
        self.steps = 0
        self.step_per_len_increase = step_per_len_increase
        self.get_header = get_headers
        
        # Iterate through all fasta files in directory
        LOGGER.info(f'Reading sequences from {fasta_data_dir}')
        #for filename in tqdm(os.listdir(fasta_data_dir)):
        file_seqs = []
        file_headers = []
            
        # Parse each fasta file
        fasta_path = os.path.join(fasta_data_dir, filename)

        if not include_dna and n > -1:
            progress_bar = tqdm(total=n, desc="Loading sequences (AA only)")
            for record in SeqIO.parse(fasta_path, "fasta"):
                seq = str(record.seq)
                seq = seq.replace('<IGS>', '-')
                self.sequences.append(seq)

                pos_ids = [i for i in range(1, len(seq) + 2)]
                self.seq_pos_ids.append(pos_ids)

                progress_bar.update(1)

                if len(self.sequences) == n:
                    break
            progress_bar.close()


        # Create progress bar with size n if specified, otherwise use default
        elif n > -1:
            progress_bar = tqdm(total=n, desc="Loading sequences")
            for record in SeqIO.parse(fasta_path, "fasta"):
                #file_seqs.append(str(record.seq))
                #file_headers.append(record.id)
                seq = str(record.seq)
                self.sequences.append(seq)
                if len(seq) > self.maxlen:
                    self.maxlen = len(seq) + 1
                pos_ids = [i for i in range(1, len(seq) + 2)]
                self.seq_pos_ids.append(pos_ids)
                
                progress_bar.update(1)
                
                if len(self.sequences) == n:
                    break
            progress_bar.close()

        else:
            if get_headers:
                self.seq_headers = []
            for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
                seq = str(record.seq)
                self.sequences.append(seq)
                if get_headers:
                    self.seq_headers.append(str(record.description))
                if len(seq) > self.maxlen:
                    self.maxlen = len(seq) + 1
                pos_ids = [i for i in range(1, len(seq) + 2)]
                self.seq_pos_ids.append(pos_ids)

        # Calculate mean and max sequence length
        seq_lengths = [len(seq) for seq in self.sequences]
        self.mean_seq_length = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
        self.max_seq_length_seen = max(seq_lengths) if seq_lengths else 0  
        self.min_seq_length = min(seq_lengths) if seq_lengths else 0

        LOGGER.info(f"Min sequence length: {self.min_seq_length}")      
        LOGGER.info(f"Mean sequence length: {self.mean_seq_length:.2f}")
        LOGGER.info(f"Max sequence length in data: {self.max_seq_length_seen}")
        if self.max_seq_len > -1:
            LOGGER.info(f"Filtering sequences to length {self.max_seq_len}")



    def set_max_seq_len(self, seq_len):
        self.max_seq_len = seq_len


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):

        if not self.mask:
            self.mask_rate = 0

        seq = list(self.sequences[idx])

        if self.max_seq_len > -1:
            #start_idx = random.sample(range(len(seq) - self.max_seq_len), 1)[0]
            contig = seq[:self.max_seq_len]

            if self.max_seq_len > -1:
                assert len(contig) <= self.max_seq_len and len(contig) > 0
        else:
            contig = seq
            assert len(contig) > 0

        num_elements_to_pick = int(len(contig) * self.mask_rate)
        picked_indices = random.sample(range(len(contig)), num_elements_to_pick)
        masked_seq = list(contig)

        for i in picked_indices: 
            masked_seq[i] = "<MASK>"

        pos_ids = [i for i in range(1, len(contig) + 3)]
        pos = self.seq_pos_ids[idx]

        if self.get_header:
            return contig, masked_seq, pos, self.seq_headers[idx]
        return contig, masked_seq, pos
        

class OGMAGDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_file, k=1, mask_rate=0.15):
        self.tok = GenomicTokenizer(vocab_file=f"og_mag_vocab_k={k}", k=k)
        self.mask_rate = mask_rate
        self.seqs = []
        self.targets = []

        with open(fasta_file, 'r') as f:
            contigs = f.readlines()

            for contig in contigs:
                #tokenized_inputs_original = self.tok._tokenize(contig)
                #tokenized_inputs, tokenized_targets = self.tok.bert_masking(tokenized_inputs_original, mlm_probability=mask_rate)
                self.seqs.append(contig)
                #self.targets.append(tokenized_targets)
                #progress_bar.update(1)

    def __len__(self):
        return(len(self.seqs))

    def __getitem__(self, idx):
        x = self.seqs[idx]
        #y = self.targets[idx]
        return x, None, None, len(x)



class MAGDataset(torch.utils.data.Dataset):

    def __init__(self, fasta_data_dir, is_mask=False, mask_rate=0.15, n=-1):

        self.tok = ProteinTokenizer()
        self.is_mask = is_mask

        self.mask = is_mask
        self.mask_rate = mask_rate
        self.sequences = []
        self.masked_seqs = []
        self.seq_pos_ids = []
        self.headers = []
        self.maxlen = 0
        
        # Iterate through all fasta files in directory
        print(f'Reading sequences from {fasta_data_dir}')
        for filename in tqdm(os.listdir(fasta_data_dir)):
            file_seqs = []
            file_headers = []
                
            # Parse each fasta file
            fasta_path = os.path.join(fasta_data_dir, filename)
            for record in SeqIO.parse(fasta_path, "fasta"):
                file_seqs.append(str(record.seq))
                file_headers.append(record.id)

            
            # mask each sequence with mask_rate and tack on special tokens
            # then combine them into a contig

            masked_contig = []
            contig = []
            seq_pos_ids = []
            for seq in file_seqs:
                if not is_mask:
                    mask_rate = 0
                
                num_elements_to_pick = int(len(seq) * mask_rate)
                picked_indices = random.sample(range(len(seq)), num_elements_to_pick)
                masked_seq = [seq[i] if i not in picked_indices else '<MASK>' for i in range(len(seq))]
                masked_seq.insert(0, '<BOS>')
                assert masked_seq[0] == '<BOS>'
                pos_ids = [i for i in range(1, len(seq) + 2)]
                masked_contig += masked_seq
                seq_pos_ids += pos_ids

                nonmasked_seq = list(seq)
                nonmasked_seq.insert(0, '<BOS>')
                assert nonmasked_seq[0] == '<BOS>'
                contig += nonmasked_seq

                if len(pos_ids) > self.maxlen:
                    self.maxlen = len(pos_ids)

            self.masked_seqs.append(masked_contig)
            self.seq_pos_ids.append(seq_pos_ids)
            self.sequences.append(contig)
            self.headers.append(filename)

            if n > -1 and len(self.sequences) == n: break


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):
        contig = self.sequences[idx]
        pos = self.seq_pos_ids[idx]
        masked_contig = None
        if self.is_mask:
            masked_contig = self.masked_seqs[idx]
        return contig, masked_contig, pos
        

class CosineWarmupScheduler:
    """
    Implements a learning rate scheduler with linear warmup followed by cosine annealing.
    Tracks steps directly rather than epochs.
    
    Args:
        optimizer: The optimizer to adjust the learning rate for
        warmup_steps: Number of steps for the warmup phase
        total_steps: Total number of training steps
        min_lr_factor: Minimum learning rate as a fraction of initial learning rate at the end of scheduling
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, 
                 min_lr_factor: float = 1e-3, steps_to_reset = 500):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_factor = min_lr_factor
        self.current_step = 0
        self.reset_steps = 0
        self.steps_to_reset = steps_to_reset
        
        # Store initial learning rates for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate and take a step"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self._compute_lr(self.base_lrs[i])
        
        self.current_step += 1
        self.reset_steps += 1
    
    def _compute_lr(self, base_lr):
        if self.current_step < self.warmup_steps:
            # Linear warmup phase
            alpha = self.current_step / self.warmup_steps
            scale_factor = alpha  # Linear increase from 0 to 1
            return base_lr * scale_factor
            
        else:
        #    # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Cap at 1.0 to handle overflow
            scale_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale_factor = scale_factor * (1.0 - self.min_lr_factor) + self.min_lr_factor
            return base_lr * scale_factor
    
    def get_lr(self):
        """Return current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_last_lr(self):
        """Return current learning rates (for compatibility with PyTorch's schedulers)"""
        return self.get_lr()

    def reset(self):
        self.current_step = 0


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.drop('Unnamed: 0', axis=1)
        df = df.dropna(subset=['ko_annot'])

        self.targets = df.ko_annot.to_list()
        self.x = df[df.columns[:-1]].values

        # Get unique targets and create mapping to integers
        unique_targets = list(set(self.targets))
        self.target_to_idx = {target: idx for idx, target in enumerate(unique_targets)}
        self.idx_to_target = {idx: target for target, idx in self.target_to_idx.items()}
        self.num_classes = len(unique_targets)
        
        # Convert targets to integer indices
        self.target_indices = [self.target_to_idx[target] for target in self.targets]


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.x[idx], self.target_indices[idx]



"""
Class strictly for the Protein OG dataset
"""
class ProtOGDataset(Dataset):
    def __init__(self, tokenizer):
        self.ds = datasets.load_dataset("tattabio/OMG_prot50")['train']
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        seq = sample['sequence']
        return self.tokenizer(seq) #, padding=True, return_tensors='pt')
    


class OMGDataset(Dataset):
    def __init__(
        self, 
        tokenizer, 
        data_subset_path="tattabio/OMG", 
        custom_mags=False, 
        load_dedup=False, 
        seq_length_max=100000, 
        output_cds=False, 
        return_offset_mapping=False,
        ):

        if load_dedup:
            self.ds = datasets.load_from_disk(data_subset_path)
        elif custom_mags:
            self.ds = datasets.load_dataset("parquet", data_files=data_subset_path)["train"].shuffle(seed=42)
            print(f"# of contigs loaded: {len(self.ds)}")
        else:
            self.ds = datasets.load_dataset(data_subset_path)["train"].shuffle(seed=42)
        self.tokenizer = tokenizer
        self.seq_length_max = seq_length_max
        self.aminos = 'ACDEFGHIKLMNPQRSTVWYBZJXUO'
        self.ntds = 'acgtn'
        self.output_cds = output_cds
        self.return_offsets = return_offset_mapping

    def __len__(self):
        return len(self.ds)


    def is_aa(self, seq):
        if seq[-2] in self.aminos:
            return True
        return False


    def get_amino_positions(self, pairs):
        aa_id = 1
        segments = []
        for segment in pairs:
            seq = segment[1]
            if self.is_aa(seq):
                this_segment = torch.full((len(seq)-2,), aa_id)
                this_segment[0] = 0 # turn off ori token
                segments.append(this_segment)
                aa_id += 1
            else:
                segments.append(torch.full((len(seq)-2,), 0))
        
        seg_ids = torch.cat(segments)
        return seg_ids


    def get_protein_coords(self, inputs):
        
        token_ids = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # 3. Group continuous uppercase tokens into protein "spans"
        protein_spans = []
        current_span = None

        for i, token in enumerate(token_ids):
            # Check if the token is a single uppercase amino acid
            if len(token) == 1 and token.isupper():
                if current_span is None:
                    # Start of a new protein
                    current_span = {"start_token": i, "end_token": i}
                else:
                    # Continuation of the current protein
                    current_span["end_token"] = i
            else:
                # We hit DNA or a special marker; close the current span if it exists
                if current_span is not None:
                    protein_spans.append(current_span)
                    current_span = None

        # If the sequence ends with a protein, close the last span
        if current_span is not None:
            protein_spans.append(current_span)

        coords = []
        for idx, span in enumerate(protein_spans):
            s, e = span["start_token"], span["end_token"]
            coords.append( (s, e))

        return coords




    def __getitem__(self, idx):
        sample = self.ds[idx]
        cds_seqs = sample["CDS_seqs"]
        cds_ids = sample['CDS_ids']
        cds_positions = sample["CDS_position_ids"]
        igs_seqs = sample["IGS_seqs"]
        igs_positions = sample["IGS_position_ids"]
        ori_cds = sample["CDS_orientations"]
        ori_cds = ["<+>" if ori else "<->" for ori in ori_cds]
        #cds_labels = sample['CDS_labels']

        for i in range(len(cds_seqs)):
            cds_seqs[i] = cds_seqs[i].replace('*', "")

        cds_pairs = [((cds_positions[i], ori_cds[i]+cds_seqs[i])) for i in range(len(cds_seqs))]
        igs_pairs = [(igs_positions[i], igs_seqs[i]) for i in range(len(igs_seqs))]
        igs_pairs_lower = [(igs_positions[i], "<->"+igs_seqs[i].lower()) for i in range(len(igs_seqs))]
        pairs = sorted(igs_pairs_lower + cds_pairs, key=lambda x: x[0])

        protein_segments = self.get_amino_positions(pairs)
        seq = "".join([pair[1] for pair in pairs])
    
        if len(seq) > self.seq_length_max:
            seq = seq[:self.seq_length_max]

        output = self.tokenizer(seq, return_tensors='pt') #, return_offset_mapping=self.return_offsets)
        output['sequence'] = seq
        output['protein_segments'] = protein_segments[:output['input_ids'].size(1)]
        
        #output['labels'] = cds_labels
        output['CDS_ids'] = cds_ids
        
        if self.return_offsets:
            output['protein_coords'] = self.get_protein_coords(output)
        if self.output_cds:
            output['CDS_seqs'] = cds_seqs
        #assert len(cds_seqs) == len(cds_labels), "Number of proteins != number of labels!"
        return output



class EsmProteinKODataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        df = pd.read_csv(csv_file)
        self.sequences = list(df.sequence)
        self.labels = list(df.KO)
        self.nu_labels = len(df.KO.unique())
        self.tokenizer = tokenizer

        print(f"# of sequences for training: {len(self.sequences)}")
        print(f"# of labels: {self.nu_labels}")

        self.label_idx = dict(zip(list(df.KO.unique()), [i for i in range(self.nu_labels)]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.tokenizer(
            self.sequences[idx],
            return_tensors='pt',
            padding=True,
            truncation=True,
        )

        ko = self.labels[idx]
        sample['labels'] = self.label_idx[ko]
        return sample



def balance_dataframe(df, label_col, n_samples, seed):
    """
    Resamples a dataframe so every label has exactly n_samples.
    If a label has < n_samples, it samples with replacement (upsampling).
    If a label has > n_samples, it samples without replacement (downsampling).
    """
    def sampler(group):
        # If we have enough samples, don't replace (downsample)
        # If we don't have enough, replace=True (upsample)
        return group.sample(n=n_samples, replace=len(group) < n_samples, random_state=seed)

    return df.groupby(label_col, group_keys=False).apply(sampler)




class ProteinKODataset(Dataset):
    def __init__(self, csv_file, tokenizer, data_seed=42, seq_per_label=-1, load_labels=False, label_map=None):
        df = pd.read_csv(csv_file)

        
        if seq_per_label > 0:
            print(f"Balancing data so we have {seq_per_label} sequences per KO")
            df = balance_dataframe(df, label_col='KO', n_samples=seq_per_label, seed=data_seed)
            #df = df.groupby('KO', group_keys=False).apply(lambda x: x.sample(seq_per_label))

        self.sequences = list(df.sequence)
        self.labels = list(df.KO)
        self.nu_labels = len(df.KO.unique())
        self.tokenizer = tokenizer

        print(f"# of sequences for training: {len(self.sequences)}")
        print(f"# of labels: {self.nu_labels}")

        if load_labels:
            assert label_map is not None, "Indicated label map LOADING but no label map path provided."
            with open(label_map, "r") as f:
                self.label_idx = json.load(f)
            print(f"Loaded label map to {label_map}")
        else:
            assert label_map is not None, "Indicated label map SAVING but no label map path provided."
            self.label_idx = dict(zip(list(df.KO.unique()), [i for i in range(self.nu_labels)]))
            with open(label_map, "w") as f:
                json.dump(self.label_idx, f, indent=4)
            print(f"Saved label map to {label_map}")
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.tokenizer(
            self.sequences[idx],
        )

        ko = self.labels[idx]
        sample['labels'] = self.label_idx[ko]
        return sample




def construct_glm_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True, use_fast=True)
    nucleotides = ['a', 't', 'c', 'g']
    ori = ["<+>", "<->"]
    tokenizer.add_special_tokens({'additional_special_tokens': nucleotides + ori})
    return tokenizer



def create_protein_batches(data_length, batch_size=8):
    sampler = torch.utils.data.RandomSampler(range(data_length))
    return list(torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False))


def pad_batch(tok_seqs, labels, max_size, pad_value=21):
    def pad_sequence(seq):
        return torch.nn.functional.pad(seq, (0, max_size - len(seq)), value=pad_value)

    batch_padded = torch.stack([pad_sequence(seq) for seq in tok_seqs])
    batch_padded_label = torch.stack([pad_sequence(label) for label in labels])

    return batch_padded, batch_padded_label



def plot_losses(loss1, loss2, loss3, label1, label2, label3, title, ylabel, xlabel, ppls=None, smoothing_window=25, save_path=None):
    """
    Plot training losses and perplexities with smoothing.
    
    Args:
        losses (list): List of loss values during training
        ppls (list, optional): List of perplexity values during training
        smoothing_window (int): Window size for moving average smoothing
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    
    # Create figure with subplots if perplexity data is provided
    if ppls:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot raw losses
    if loss1: ax1.plot(loss1, 'b-', alpha=0.7, label=label1)
    if loss2: ax1.plot(loss2, 'r-', alpha=0.7, label=label2)
    if loss3: ax1.plot(loss3, 'c-', alpha=0.7, label=label3)

    # Add labels and title for loss plot
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel(xlabel)
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

