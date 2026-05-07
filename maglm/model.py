from maglm.config import PLMConfig
from torch import nn, arange
import torch.nn.functional as F
import math
import torch
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP
from torch.utils.checkpoint import checkpoint_sequential
from functools import partial
from typing import Optional, Tuple, Union
from maglm.config import (
    BiMambaSequenceClassifierConfig,
    BiMambaConfig,
    ESMConfig,
    gLMConfig,
)

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
from transformers.modeling_outputs import BaseModelOutput







# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)



'''
https://github.com/kuleshov-group/caduceus/blob/main/caduceus/modeling_caduceus.py
'''
class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[str] = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            #if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
            #    self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
            #    self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
            #    self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
            #    self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        return out



class BiMambaBlock(torch.nn.Module):

    def __init__(self, d_model, d_inter, d_state, d_conv, expand, bidirectional_strategy='add'):
        super().__init__()

        mixer_cls = partial(
            BiMambaWrapper,
            bidirectional=True,
            bidirectional_strategy=bidirectional_strategy,
            #d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_init="random",
        )

        norm_cls = partial(torch.nn.RMSNorm, eps=1e-5)

        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_inter,
            out_features=d_model,
            activation=F.silu,
        )

        self.mamba_block_forward = Block(
            dim=d_model,
            mixer_cls=mixer_cls,
            mlp_cls=mlp_cls,
            norm_cls=norm_cls,
            fused_add_norm=False,
        )

    def forward(self, x, r_f=None): #, r_rev=None):
        h, residual = self.mamba_block_forward(x, r_f)
        return h + residual, residual




class BiMambaModel(torch.nn.Module):
    def __init__(
        self, layers, d_model, d_inter, vocab_size, d_state=8, d_conv=4, expand=2, dropout=0.0, bidirectional_strategy='add',
    ):
        super().__init__()
        self.lm_norm = torch.nn.RMSNorm(d_model, eps=1e-5)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.mamba_blocks = torch.nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_inter=d_inter,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bidirectional_strategy=bidirectional_strategy,
                )
                for _ in range(layers)
            ]
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=layers,
                n_residuals_per_layer=2,
            )
        )

    def forward(self, x):
        h = x 
        residual = None
        for block in self.mamba_blocks:
            h, residual = block(h, residual)
        h_norm = self.lm_norm(h)
        lm_output = self.lm_head(h_norm)
        return lm_output, h_norm, h



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



"""Sequence classifier wrapper to use with pretrained BiMamba models."""
class BiMambaForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        backbone_config = BiMambaConfig.from_pretrained(config.model_backbone_ckpt)

        self.backbone = BiMambaMLM.from_pretrained(config.model_backbone_ckpt, config=backbone_config)
        self.clf_head_norm = torch.nn.LayerNorm(backbone_config.d_model)
        self.clf_head = MLP(input_size=backbone_config.d_model, output_size=config.nu_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')


    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        _, h = self.backbone(input_ids)

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
        with torch.no_grad():
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


    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights """
        output_embeddings.weight = input_embeddings.weight
        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias = input_embeddings.bias

    def tie_weights(self, *args, **kwargs):
        """
        Physically link the tensors in memory.
        """
        
        # Tie Mamba blocks
        for block in self.bimamba_layer.mamba_blocks:
            mixer = block.mamba_block_forward.mixer
            self._tie_or_clone_weights(mixer.mamba_rev.in_proj, mixer.mamba_fwd.in_proj)
            self._tie_or_clone_weights(mixer.mamba_rev.out_proj, mixer.mamba_fwd.out_proj)


    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None, output_hidden_states=True): #**kwargs):
        
        h = self.tok_embedding(input_ids) 

        if self.include_token_type:
            h = h * self.tok_type_embedding(token_type_ids)
        
        if output_hidden_states:
            all_hidden_states = (h,)
            res = None
            for layer in self.bimamba_layer.mamba_blocks:
                h, res = layer(h, res)
                all_hidden_states = all_hidden_states + (h,)

            return BaseModelOutput(
                last_hidden_state=h,
                hidden_states=all_hidden_states,
            )

        else:
            logits, h_norm, h = self.bimamba_layer(h) #.squeeze(0))
            if labels is not None: 
                labels = labels.squeeze(0)

                loss = self.loss_fn(
                    logits.view(-1, logits.shape[2]), 
                    labels.view(-1)
                    ).mean()
                return (loss, logits)
            return (logits, h_norm, h)