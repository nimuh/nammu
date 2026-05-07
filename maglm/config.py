from dataclasses import dataclass
import torch
from transformers import PretrainedConfig


@dataclass
class PLMConfig:
    block_size: int = 1024
    vocab_size: int = 20
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 256



@dataclass
class BioMambaConfig:
    layers: int = 1
    d_model: int = 64
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    vocab_size: int = 25
    dropout: float = 0.1
    cuda_rank: int = 0
    model_saved_name: str = None


@dataclass
class TrainConfig:
    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 1
    nsamples: int = 100
    steps: int = 1000


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