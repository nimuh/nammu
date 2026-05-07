import argparse
import torch
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
    EsmForSequenceClassification,
    EsmTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
from evaluate import load
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from maglm.model import BiMambaModel
import datasets
from tqdm import tqdm
from prettytable import PrettyTable
import json
import os
import seaborn as sns
import rmm
from rmm.allocators.torch import rmm_torch_allocator
from accelerate import Accelerator
from accelerate.utils import convert_model
from torch.optim import Adafactor, AdamW
from transformers.trainer_utils import get_last_checkpoint
import gc
from safetensors.torch import load_file




# USE THIS IF RUNNING ON GRACE HOPPER 
#############################################################################
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
#############################################################################

# some globals
accuracy = load("accuracy")
matthews_metric = load("matthews_correlation")
f1_metric = load("f1")
precision_metric = load("precision")
recall_metric = load('recall')
conf_mat_metric = load("confusion_matrix")



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    acc = accuracy.compute(predictions=predictions, references=labels)
    mcc = matthews_metric.compute(references=labels, predictions=predictions)    
    f1 = f1_metric.compute(references=labels, predictions=predictions, average='weighted')
    return {"accuracy": acc, "MCC": mcc, "F1": f1, 'precision': precision, 'recall': recall}



def print_model_size(model):
    total_params = model.num_parameters()
    trainable_params = model.num_parameters(only_trainable=True)
    
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"Trainable Ratio: {trainable_params/total_params:.2%}")


"""
Little function for printing our parameters/layer :D
"""
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


def check_bimamba_load(model):
    """Verify that bidirectional weight tying was loaded correctly."""
    # Access the first block to check
    block = model.bimamba_layer.mamba_blocks[0].mamba_block_forward.mixer

    # Get the forward and reverse weights
    fwd_weight = block.mamba_fwd.in_proj.weight
    rev_weight = block.mamba_rev.in_proj.weight

    # Check if they are physically the same memory object
    is_tied_memory = (fwd_weight.data_ptr() == rev_weight.data_ptr())

    # Check if they have the exact same values
    are_values_equal = torch.equal(fwd_weight, rev_weight)

    print(f" - Memory Tied: {is_tied_memory}")
    print(f" - Values Equal: {are_values_equal}")

    if not are_values_equal:
        raise ValueError("CRITICAL: Reverse weights were not loaded correctly! Stop training.")
    else:
        print("SUCCESS: Weights are tied and loaded. The 'Missing Keys' warning is a false alarm.")


# CONFIG CLASSES

class BiMambaSequenceClassifierConfig(PretrainedConfig):
    model_type = "bimamba_classifier"
    def __init__(
        self,
        model_backbone_ckpt='bimamba_clf',
        nu_labels=2,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.model_backbone_ckpt = model_backbone_ckpt
        self.nu_labels = nu_labels




class BiMambaConfig(PretrainedConfig):
    model_type = "bimamba_mlm"
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 8,
        d_inter: int = 768,
        nlayers: int = 7,
        vocab_size=37,
        bidirectional_strategy='add',
        tie_weights=True,
        include_token_type=True,
        max_seq_len=100000000,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.tie_word_embeddings = True
        self.d_model = d_model
        self.d_state = d_state
        self.d_inter = d_inter,
        self.nlayers = nlayers,
        self.vocab_size = vocab_size
        self.include_token_type = include_token_type
        self.max_seq_len = max_seq_len,
        self.bidirectional_strategy = bidirectional_strategy
        self.tie_weights = tie_weights




class gLMConfig(PretrainedConfig):
    model_type = "glm_classifier"
    def __init__(self, nu_labels=7000, glm_ckpt='tattabio/gLM2_150M', **kwargs):
        super().__init__(**kwargs)
        self.nu_labels = nu_labels
        self.glm_ckpt = glm_ckpt
        self.d_model = 640


class ESMConfig(PretrainedConfig):
    model_type = "esm2"
    def __init__(self, nu_labels=7000, esm_ckpt='facebook/esm2_t6_8M_UR50D', freeze_backbone=False, **kwargs):
        super().__init__(**kwargs)
        self.nu_labels = nu_labels
        self.esm_ckpt = esm_ckpt
        self.d_model = 640
        self.freeze_backbone = freeze_backbone

"""Generic classification head."""
class MLP(torch.nn.Module):
    def __init__(self, output_size, input_size):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 1024)
        self.linear2 = torch.nn.Linear(1024, 2048)
        self.linear3 = torch.nn.Linear(2048, 4096)
        self.linear4 = torch.nn.Linear(4096, output_size)

    def forward(self, x):
        h = self.linear1(x)
        h = self.linear2(F.relu(h))
        h = self.linear3(F.relu(h))
        h = self.linear4(F.relu(h))
        return h



class MLMHead(torch.nn.Module):
    def __init__(self, output_size, input_size):
        super(MLMHead, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)




class BiMambaForSequenceClassification(PreTrainedModel):
    """Sequence classification model using a pretrained BiMamba backbone.

    This wrapper adds a classification head on top of BiMambaMLM for sequence classification tasks.

    Attributes:
        backbone (nn.Module): The BiMamba MLM model as feature extractor.
        clf_head_norm (nn.LayerNorm): Layer normalization before classification head.
        clf_head (nn.Module): MLP classification head.
        loss_fn (nn.Module): Cross-entropy loss for training.
    """

    def __init__(self, config):
        """Initialize the sequence classification model.

        Args:
            config (BiMambaSequenceClassifierConfig): Configuration object.
        """
        super().__init__(config)

        backbone_config = BiMambaConfig.from_pretrained(config.model_backbone_ckpt)
        # Load frozen or trainable backbone; store normalization and classifier head.
        self.backbone = BiMambaMLM.from_pretrained(
            config.model_backbone_ckpt,
            config=backbone_config,
        )
        self.clf_head_norm = torch.nn.LayerNorm(backbone_config.d_model)
        self.clf_head = MLP(
            input_size=backbone_config.d_model,
            output_size=config.nu_labels
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        """Forward pass for sequence classification.

        Args:
            input_ids (Tensor): Token IDs.
            labels (Tensor): True labels.
            attention_mask (Tensor): Attention mask for input.
            token_type_ids (Tensor): Segment IDs (unused).

        Returns:
            Tuple(Tensor, Tensor) or Tensor: Loss and logits if labels are provided, else logits.
        """
        _, h = self.backbone(input_ids)
        h = self.mean_mask_pooling(h, attention_mask)
        # h = torch.mean(h, dim=1)  # Optionally, vanilla mean pooling (not mask-aware)
        h = self.clf_head_norm(h)

        preds = self.clf_head(h)
        if labels is not None:
            loss = self.loss_fn(preds, labels)
            return (loss, preds)
        return preds

    def mean_mask_pooling(self, h, attention_mask):
        """Pool embeddings using the attention mask for mean calculation.

        Args:
            h (Tensor): Hidden states (batch_size, seq_len, hidden_dim).
            attention_mask (Tensor): Mask indicating valid tokens.

        Returns:
            Tensor: Pooled embeddings for each sequence in the batch.
        """
        inputs_mask_expanded = attention_mask.unsqueeze(-1).expand(h.size()).float()
        sum_embeddings = torch.sum(h * inputs_mask_expanded, 1)
        sum_mask = torch.clamp(inputs_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        return pooled_output




"""Wrapper class over ESM-2 for sequence classification."""
class ESM2ForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = AutoModel.from_pretrained(
            config.esm_ckpt, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            )
        self.clf_head_norm = torch.nn.LayerNorm(config.d_model)
        self.clf_head = MLP(input_size=config.d_model, output_size=config.nu_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.freeze_backbone = config.freeze_backbone


    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):

        assert attention_mask is not None

        if self.freeze_backbone:
            with torch.no_grad():
                h = self.backbone(input_ids, output_hidden_states=True).last_hidden_state
        else:
            h = self.backbone(input_ids, output_hidden_states=True).last_hidden_state

        h = self.mean_mask_pooling(h, attention_mask)
        h = self.clf_head_norm(h)

        preds = self.clf_head(h)
        if labels is not None:
            loss = self.loss_fn(preds, labels)
            return (loss, preds)
        return preds

    def mean_mask_pooling(self, h, attention_mask):
        assert attention_mask is not None, "No attention mask given to do mean pooling!"
        inputs_mask_expanded = attention_mask.unsqueeze(-1).expand(h.size()).float()
        sum_embeddings = torch.sum(h * inputs_mask_expanded, 1)
        sum_mask = torch.clamp(inputs_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        return pooled_output




"""Wrapper around gLM2 for sequence classification tasks."""
class gLM2ForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = AutoModel.from_pretrained(config.glm_ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.clf_head_norm = torch.nn.LayerNorm(config.d_model)
        self.clf_head = MLP(input_size=config.d_model, output_size=config.nu_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')


    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_cudnn=False):
            h = self.backbone(input_ids, output_hidden_states=True).last_hidden_state

        h = self.mean_mask_pooling(h, attention_mask)
        h = self.clf_head_norm(h)

        preds = self.clf_head(h)
        if labels is not None:
            loss = self.loss_fn(preds, labels)
            return (loss, preds)
        return preds


    def mean_mask_pooling(self, h, attention_mask):
        inputs_mask_expanded = attention_mask.unsqueeze(-1).expand(h.size()).float()
        sum_embeddings = torch.sum(h * inputs_mask_expanded, 1)
        sum_mask = torch.clamp(inputs_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        return pooled_output


class BiMambaMLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.include_token_type = config.include_token_type
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.bidirectional_strategy = config.bidirectional_strategy

        # need to check for nested lists in the config (not sure why this is happening with hf)
        self.nlayers = config.nlayers[0]
        if isinstance(self.nlayers, list):
            self.nlayers = self.nlayers[0]
            if isinstance(self.nlayers, list): self.nlayers = self.nlayers[0]

        self.max_seq_len = config.max_seq_len[0]
        if isinstance(self.max_seq_len, list):
            self.max_seq_len = self.max_seq_len[0]
            if isinstance(self.max_seq_len, list): self.max_seq_len = self.max_seq_len[0]

        self.d_inter = config.d_inter[0]
        if isinstance(self.d_inter, list):
            self.d_inter = self.d_inter[0]
            if isinstance(self.d_inter, list): self.d_inter = self.d_inter[0]

        self.vocab_size = config.vocab_size
        self.tok_embedding = torch.nn.Embedding(self.vocab_size, self.d_model)

        if self.include_token_type:
            self.tok_type_embedding = torch.nn.Embedding(3, self.d_model)

        self.bimamba_layer = BiMambaModel(
            layers=self.nlayers,
            d_state=self.d_state,
            bidirectional_strategy=self.bidirectional_strategy,
            d_model=self.d_model,
            d_inter=self.d_inter,
            vocab_size=self.vocab_size,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.post_init()

        if config.tie_weights:
            self.tie_weights()


    def get_input_embeddings(self):
        return self.tok_embedding

    def set_input_embeddings(self, value):
        self.tok_embedding = value

    def get_output_embeddings(self):
        return self.bimamba_layer.lm_head

    def set_output_embeddings(self, vocab_size):
        self.bimamba_layer.lm_head = torch.nn.Linear(self.d_model, vocab_size)


    @property
    def _tied_weights_keys(self):
        """Dynamically generate the list of all tied weight keys for safetensors."""
        # Start with the standard embedding/head tie
        tied_keys = []
       
        # Add all the BiMamba reverse projection weights
        # We assume the 'forward' is the master and 'reverse' is the alias
        for i in range(len(self.bimamba_layer.mamba_blocks)):
            base = f"bimamba_layer.mamba_blocks.{i}.mamba_block_forward.mixer"
            tied_keys.append(f"{base}.mamba_rev.in_proj.weight")
            tied_keys.append(f"{base}.mamba_rev.out_proj.weight")
            
        return tied_keys


    def tie_weights(self):
        """
        Physically link the tensors in memory.
        """
        
        # Tie Mamba blocks
        for block in self.bimamba_layer.mamba_blocks:
            mixer = block.mamba_block_forward.mixer
            self._tie_or_clone_weights(mixer.mamba_rev.in_proj, mixer.mamba_fwd.in_proj)
            self._tie_or_clone_weights(mixer.mamba_rev.out_proj, mixer.mamba_fwd.out_proj)


    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        
        if input_ids.size(-1) > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            labels = labels[:, :self.max_seq_len]
            token_type_ids = token_type_ids[:, :self.max_seq_len]

        h = self.tok_embedding(input_ids) 

        if self.include_token_type:
            h = h * self.tok_type_embedding(token_type_ids)

        logits, h_norm, h = self.bimamba_layer(h) #.squeeze(0))
        if labels is not None: 
            labels = labels.squeeze(0)

            loss = self.loss_fn(
                logits.view(-1, logits.shape[2]), 
                labels.reshape(-1)
                ).mean()
            return (loss, logits)
        return (logits, h_norm, h)




class OMGDataset(Dataset):
    def __init__(self, tokenizer, data_subset_path="tattabio/OMG", load_dedup=False, seq_length_max=100000, remove_igs=False, skip_n=-1):
        if load_dedup:
            self.ds = datasets.load_from_disk(data_subset_path)
        else:
            self.ds = datasets.load_dataset(data_subset_path)["train"].shuffle(seed=42)
        
        if skip_n > 0:
            print(f"Skipping {skip_n} samples...")
            self.ds = self.ds.select(range(skip_n, len(self.ds)))

        self.tokenizer = tokenizer
        self.seq_length_max = seq_length_max
        self.remove_igs = remove_igs

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        cds_seqs = sample["CDS_seqs"]
        cds_positions = sample["CDS_position_ids"]
        igs_seqs = sample["IGS_seqs"]
        igs_positions = sample["IGS_position_ids"]
        ori_cds = sample["CDS_orientations"]
        ori_cds = ["<+>" if ori else "<->" for ori in ori_cds]

        cds_pairs = [((cds_positions[i], ori_cds[i]+cds_seqs[i])) for i in range(len(cds_seqs))]

        if self.remove_igs:
            igs_pairs_lower = [(igs_positions[i], "<sep>") for i in range(len(igs_seqs))]
        else:
            igs_pairs_lower = [(igs_positions[i], "<->"+igs_seqs[i].lower()) for i in range(len(igs_seqs))]

        pairs = sorted(igs_pairs_lower + cds_pairs, key=lambda x: x[0])
        seq = "".join([pair[1] for pair in pairs])
        if len(seq) > self.seq_length_max:
            seq = seq[:self.seq_length_max]
        return self.tokenizer(seq)



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



class ProteinKODataset(Dataset):
    def __init__(self, csv_file, tokenizer, data_seed=42, seq_per_label=-1, load_labels=False, label_map=None):
        df = pd.read_csv(csv_file)

        
        if seq_per_label > 0:
            print(f"Balancing data so we have {seq_per_label} sequences per KO")
            df = balance_dataframe(df, label_col='KO', n_samples=seq_per_label, seed=data_seed)

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



def construct_glm_tokenizer(only_mask_proteins=True):
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)
    if only_mask_proteins:
        nucleotides = ['a', 't', 'c', 'g']
        ori = ["<+>", "<->"]
        tokenizer.add_special_tokens({'additional_special_tokens': nucleotides + ori})
        return tokenizer
    return tokenizer



def bimamba_mlm_multi(
    model_dir,
    mamba_layers,
    ds,
    eval_ds,
    tokenizer,
    d_model=256,
    d_inter=256,
    bs=8,
    grad_accum=1,
    total_opt_steps=10000,
    max_seq_len=50000,
    d_state=8,
    optimizer='adamw',
    include_token_type=False,
    bidirectional_strategy='add',
    save_every=5000,
    resume_ckpt=False,
    ckpt=None,
    mlm_probability=0.15,
    lr_scheduler_type='cosine',
    lr=1e-3,
    lr_scheduler_args=None,
    ):

    
    ds.seq_length_max = max_seq_len
    print(f"Truncating sequences to {ds.seq_length_max}")
    print(f"Tokenizer size: {len(tokenizer)}")
    print(f"CPU count: {len(os.sched_getaffinity(0))}")


    # this should also handle batching directly
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True,
        mlm_probability=mlm_probability,
        return_tensors='pt',
    )

    warm_up_steps = int(0.1 * total_opt_steps)

    tr_args = TrainingArguments(
        output_dir=model_dir,
        data_seed=2222,
        learning_rate=lr,
        weight_decay=0.1,
        max_steps=total_opt_steps,
        overwrite_output_dir=False,
        warmup_steps=warm_up_steps,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_args,
        logging_strategy='steps',
        logging_steps=100,
        auto_find_batch_size=False,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        local_rank=-1,
        include_num_input_tokens_seen=True,
        disable_tqdm=False,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=save_every,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=0.5,
        bf16=True,
        fp16=False,
        gradient_accumulation_steps=grad_accum,
        resume_from_checkpoint=False,
    )

    if resume_ckpt:
        print(f"Resuming training...")
        last_checkpoint = None
        if os.path.isdir(tr_args.output_dir):
            last_checkpoint = get_last_checkpoint(tr_args.output_dir)
            print(f"Found last checkpoint: {last_checkpoint}")
        config = BiMambaConfig.from_pretrained(last_checkpoint)
        model = BiMambaMLM.from_pretrained(last_checkpoint, config=config)
        model.max_seq_len = max_seq_len
        print(f"Set context length = {model.max_seq_len}")
    elif ckpt:
        print(f"Loading checkpoint: {ckpt}")
        config = BiMambaConfig.from_pretrained(ckpt)
        model = BiMambaMLM.from_pretrained(ckpt, config=config)
        model.max_seq_len = max_seq_len
        print(f"Set context length = {model.max_seq_len}")
    else:
        config = BiMambaConfig(
            d_model=d_model,
            d_inter=d_inter,
            d_state=d_state,
            max_seq_len=max_seq_len,
            bidirectional_strategy=bidirectional_strategy,
            include_token_type=include_token_type,
            nlayers=mamba_layers,
        )
        model = BiMambaMLM(config=config)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count}")

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    check_bimamba_load(model)

    if resume_ckpt:
        print(f"Resuming training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print('Starting training...')
        print(f"Training on {len(trainer.train_dataset)} sequences")
        trainer.train()




def plot_saved_loss(filepath, figname):
    loaded_loss = np.load(filepath)
    plot_running_average(loaded_loss, figname)



def plot_training_loss(checkpoint_paths, figname, labels=None, title="Training Loss Comparison"):
    """
    Plots training loss from multiple Hugging Face Trainer checkpoints.
    
    Args:
        checkpoint_paths (list): List of strings pointing to checkpoint directories.
                                 e.g., ["./run1/checkpoint-500", "./run2/checkpoint-1000"]
        labels (list, optional): Custom names for the legend. If None, uses folder names.
        title (str): Title of the plot.
    """
    # Set a "pretty" theme
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))
    
    all_data = []

    # 1. Loop through checkpoints and load data
    for i, path in enumerate(checkpoint_paths):
        json_path = os.path.join(path, "trainer_state.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: Could not find {json_path}")
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Extract history
        history = data.get("log_history", [])
        
        # Filter for training loss (ignore eval logs which have 'eval_loss')
        # We look for entries that have 'loss' and 'step'
        train_logs = [entry for entry in history if 'loss' in entry]
        
        if not train_logs:
            print(f"No training loss data found in {path}")
            continue

        # Create a DataFrame for this run
        run_label = labels[i] if labels and i < len(labels) else path.split('/')[-1]
        
        df = pd.DataFrame(train_logs)
        df['Run'] = run_label
        all_data.append(df)

    if not all_data:
        print("No valid data found to plot.")
        return

    # 2. Combine and Plot
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Create the plot
    # We use 'step' for X-axis, but you could switch to 'epoch' if preferred
    sns.lineplot(
        data=final_df, 
        x="step", 
        y="loss", 
        hue="Run", 
        palette="viridis", 
        linewidth=2.5
    )

    # 3. Beautify
    plt.title(title, fontsize=20, weight='bold', pad=20)
    plt.xlabel("Training Steps", fontsize=14, labelpad=10)
    plt.ylabel("Loss", fontsize=14, labelpad=10)
    plt.legend(title="Checkpoints", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Optional: Log scale if loss drops massively (common in initial training)
    # plt.yscale('log') 
    
    plt.tight_layout()
    plt.savefig(figname)



def subset_data(ds):
    idxs_train = []
    idxs_eval_1 = []
    idxs_eval_2 = []
    idxs_eval_3 = []

    with tqdm(total=len(ds), desc="Processing Samples") as pbar:
        for i in range(len(ds)):
            sample = ds[i]
            contig_len = sample['input_ids'].size(1)
            if 8000 <= contig_len < 32000:
                idxs_train.append(i)
            elif 32000 <= contig_len < 52000:
                idxs_eval_1.append(i)
            elif 52000 <= contig_len < 72000:
                idxs_eval_2.append(i)
            elif 72000 <= contig_len < 92000:
                idxs_eval_3.append(i)
            pbar.update(1)
        filtered_ds = Subset(ds, idxs_train)

    print(f"Filtered train data size: {len(filtered_ds)}")
    np.save("../data/filtered_indices_train_OG_8K_32K.npy", np.array(idxs_train))
    np.save("../data/filtered_indices_eval_OG_32K_52K.npy", np.array(idxs_eval_1))
    np.save("../data/filtered_indices_eval_OG_52K_72K.npy", np.array(idxs_eval_2))
    np.save("../data/filtered_indices_eval_OG_72K_92K.npy", np.array(idxs_eval_3))



def load_subset(ds, subset_idxs_file):
    loaded_indices = np.load(subset_idxs_file)
    loaded_indices = [int(i) for i in loaded_indices]
    filtered_ds = Subset(ds, loaded_indices)
    print(f"Data subset for {subset_idxs_file}: {len(filtered_ds)}")
    return filtered_ds
    


def plot_training_losses():
    paths = [
        'esm_finetune_ko_test/checkpoint-2860',
        'glm2_finetune_ko_test/checkpoint-2860',
        'bimbamba_finetune_plm_contig_ko_test/checkpoint-2860',
        'bimbamba_finetune_plm_ko_test/checkpoint-2860',
        ]

    plot_training_loss(
        checkpoint_paths=paths,
        figname='../results/ko_test_glm2_esm_bimamba_contig_training_loss.png',
        labels=['ESM2-150M (frozen)', 'gLM2-150M', 'BiMamba-PLM-Contig-102M', 'BiMamba-PLM-102M'],
    )




def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for NAMMU training."""
    parser = argparse.ArgumentParser(
        description="NAMMU: Bidirectional Mamba language model for genomic contigs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_path", type=str, default="tattabio/OMG",
                        help="HuggingFace dataset name or local path to an OMG-format dataset.")
    parser.add_argument("--load_dedup", action="store_true",
                        help="Load a deduplicated dataset from disk via load_from_disk.")
    parser.add_argument("--remove_igs", action="store_true",
                        help="Replace intergenic sequences with <sep> tokens.")
    parser.add_argument("--max_seq_len", type=int, default=100000,
                        help="Maximum contig length in tokens (longer contigs are truncated).")
    parser.add_argument("--skip_n", type=int, default=-1,
                        help="Skip the first N samples (useful for curriculum training).")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=1024,
                        help="Hidden dimension of the BiMamba model.")
    parser.add_argument("--d_inter", type=int, default=768,
                        help="Intermediate dimension of the GatedMLP blocks.")
    parser.add_argument("--d_state", type=int, default=8,
                        help="SSM state dimension.")
    parser.add_argument("--nlayers", type=int, default=18,
                        help="Number of BiMamba blocks.")
    parser.add_argument("--bidirectional_strategy", type=str, default="add",
                        choices=["add", "ew_multiply"],
                        help="How to combine forward and reverse SSM outputs.")

    # Checkpointing
    parser.add_argument("--model_dir", type=str, default="models/bimamba_contig",
                        help="Output directory for saved checkpoints.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a specific checkpoint to continue training from.")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from the last checkpoint in --model_dir.")
    parser.add_argument("--save_every", type=int, default=2500,
                        help="Save a checkpoint every N optimizer steps.")

    # Optimization
    parser.add_argument("--total_steps", type=int, default=600000,
                        help="Total number of optimizer steps.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-4,
                        help="Minimum learning rate (used by cosine schedulers).")
    parser.add_argument("--bs", type=int, default=2,
                        help="Per-device train batch size.")
    parser.add_argument("--grad_accum", type=int, default=64,
                        help="Gradient accumulation steps.")
    parser.add_argument("--mlm_probability", type=float, default=0.30,
                        help="Fraction of tokens masked during MLM pre-training.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr",
                        help="Learning rate scheduler type (HuggingFace scheduler name).")

    return parser.parse_args()


def main() -> None:
    """Entry point: build dataset, model, and launch training."""
    args = parse_args()

    tokenizer = construct_glm_tokenizer(only_mask_proteins=False)

    ds = OMGDataset(
        data_subset_path=args.data_path,
        tokenizer=tokenizer,
        load_dedup=args.load_dedup,
        remove_igs=args.remove_igs,
        skip_n=args.skip_n,
    )

    bimamba_mlm_multi(
        model_dir=args.model_dir,
        ckpt=args.ckpt,
        resume_ckpt=args.resume,
        ds=ds,
        eval_ds=None,
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        total_opt_steps=args.total_steps,
        mamba_layers=args.nlayers,
        d_model=args.d_model,
        d_inter=args.d_inter,
        d_state=args.d_state,
        include_token_type=False,
        bs=args.bs,
        grad_accum=args.grad_accum,
        bidirectional_strategy=args.bidirectional_strategy,
        max_seq_len=args.max_seq_len,
        save_every=args.save_every,
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_args={"min_lr": args.min_lr},
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
