
import transformers
import itertools
import json
from transformers import PreTrainedTokenizer
from typing import List, Optional
import random
import os
import torch


def build_genomic_type_vocab():
    nucleotides = ['a', 'c', 'g', 't', 'n']
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY") 
    
    dna_kmers = []
    k = 1
    for length in range(1, k + 1):
      dna_kmers += [''.join(p) for p in itertools.product(nucleotides[:-1], repeat=length)]    
    specials = ["<PAD>", "<UNK>", "<MASK>", "<DNA_START>", "<PROT_START>", "<+>", "<->"]
    
    vocab = {}
    ntd_type = 0
    aa_type = 1
    special_type = 2

    for token in specials + dna_kmers + amino_acids:
        if token in dna_kmers:
            vocab[token] = ntd_type
        elif token in amino_acids:
            vocab[token] = aa_type
        else:
            vocab[token] = special_type
    return vocab

def build_genomic_vocab(k=6):
    # 1. Define base components
    nucleotides = ['a', 'c', 'g', 't', 'n']
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY") 
    
    # 2. Generate K-mers (e.g., 'AAAAAA', 'AAAAAC'...)
    dna_kmers = []
    for length in range(1, k + 1):
      dna_kmers += [''.join(p) for p in itertools.product(nucleotides[:-1], repeat=length)]
    
    
    # 3. Define Special Tokens (Crucial for HF)
    # <PAD>: Padding
    # <UNK>: Unknown token
    # <CLS>: Start of sequence (or global state)
    # <SEP>: Separator
    # <MASK>: For MLM
    specials = ["<PAD>", "<UNK>", "<MASK>", "<DNA_START>", "<PROT_START>", "<+>", "<->"]
    
    # 4. Create the mapping
    # Order: Specials -> DNA K-mers -> Protein Residues
    vocab = {}
    idx = 0
    for token in specials + dna_kmers + amino_acids:
        vocab[token] = idx
        idx += 1

    return vocab



class GenomicTokenizer(PreTrainedTokenizer):
    def __init__(
        self, 
        vocab_file, 
        k=6, 
        unk_token="<UNK>", 
        pad_token="<PAD>", 
        mask_token="<MASK>",
        **kwargs
    ):
        # 1. Load the vocabulary

        print(f"constructing genomic tokenizer with k={k}")
        try:
            with open(vocab_file, encoding="utf-8") as f:
                self.vocab = json.load(f)

        except FileNotFoundError:
            print(f"vocab file not found. building it...")
            vocab_dict = build_genomic_vocab(k=k)

            with open(vocab_file, "w") as f:
                json.dump(vocab_dict, f)

            with open(vocab_file, encoding="utf-8") as f:
                self.vocab = json.load(f)

        self.type_vocab = build_genomic_type_vocab()

        print(f"Vocab size={len(self.vocab)}")
        print(f"Vocab type size={len(self.type_vocab)}")

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.k = k
        
        # 2. Initialize parent class
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab


    def _tokenize(self, text):
        """
        The critical logic. 
        We assume the input text is formatted with tags to indicate modality:
        "<DNA_START> ATGCGCTAG... <PROT_START> MKL..."
        """
        tokens = []
        # Simple splitting by space to find tags vs sequence chunks
        # Note: In a real pipeline, you might handle this more robustly
        parts = text.split()
        
        current_mode = None
        
        for part in parts:
          if part in ["<DNA_START>", "<PROT_START>", "<+>", "<->", "<CLS>", "<SEP>", "<MASK>"]:
                tokens.append(part)
                if part == "<DNA_START>":
                    current_mode = "DNA"
                elif part == "<PROT_START>":
                    current_mode = "PROT"
          else:
              # If we are in DNA mode, apply K-mer stride
              if current_mode == "DNA":
                  # Ensure the part is divisible by k or handle padding
                  # Here we just stride
                  for i in range(0, len(part), self.k):
                      kmer = part[i : i + self.k]
                      tokens.append(kmer)
                      
              # If we are in Protein mode, apply Character split
              elif current_mode == "PROT":
                  tokens.extend(list(part))
              
              # If no mode set (or special tokens), keep as is
              else:
                  tokens.append(part)
                    
        return [self._convert_token_to_id(token) for token in tokens] #, [self._convert_token_to_type(token) for token in tokens]


    def bert_masking(self, tokens, mlm_probability=0.15):

        inputs = torch.Tensor(tokens.copy()).long()
        targets = torch.full_like(inputs, self.pad_token_id).long()

        contig_len = len(tokens)
        contig_idxs = list(range(0, contig_len))
        pos_to_mask = max(1, int(contig_len * mlm_probability))
        selected_pos_to_mask = random.sample(contig_idxs, pos_to_mask)

        for t in selected_pos_to_mask:
            p = random.random()
            if p <= 0.10:
                inputs[t] = inputs[t].item()
            elif p <= 0.20:
                inputs[t] = random.choice(list(self.vocab.values())) 
            else:
                inputs[t] = self.mask_token_id
            targets[t] = tokens[t]

        token_types = torch.Tensor([self._convert_token_to_id(token) for token in self.decode(inputs)]).long()

        assert len(inputs) == len(targets), "Mismatch between inputs and targets!"
        return inputs, token_types, targets



    """
    This scheme only masks proteins but will mask the amino acids the same
    way BERT does.
    """

    def mask_proteins(self, tokens, mlm_probability=0.15):
        inputs = []
        targets = []
        do_mask = False
        for i in range(len(tokens)):

            # case where you are at the start of a CDS segment
            if self.ids_to_tokens[tokens[i]] == "<PROT_START>":
                inputs.extend([tokens[i], tokens[i+1]])
                targets.extend([self.pad_token_id] * 2)

                i = i + 2 # skip orientation tokens

                protein_len = 0
                protein_idxs = None
                end_of_contig = None
                for j in range(i, len(tokens)):
                    if self.ids_to_tokens[tokens[j]] == "<DNA_START>":
                        protein_len = len(tokens[i:j])
                        protein_idxs = list(range(i, j))
                        break
                    protein_len = len(tokens[i:j])
                    protein_idxs = list(range(i, j+1))
                
                if j == len(tokens) - 1: end_of_contig = j + 1
                else: end_of_contig = j
                    

                pos_to_mask = max(1, int(protein_len * mlm_probability))
                selected_pos_to_mask = random.sample(protein_idxs, pos_to_mask)
                
                # iterate across protein here
                for t in range(i, end_of_contig):
                    if t in selected_pos_to_mask:
                        p = random.random()
                        if p < 0.05:
                            inputs.append(tokens[t])
                        elif p < 0.10:
                            inputs.append(random.choice(list(self.vocab.values())))
                        else:
                            inputs.append(self.mask_token_id)
                        targets.append(tokens[t])
                    else:
                        inputs.append(tokens[t])
                        targets.append(self.pad_token_id)
                i = j
                continue

            # case where we are at the start of an IGS segment
            elif self.ids_to_tokens[tokens[i]] == "<DNA_START>":
                inputs.extend([tokens[i], tokens[i+1]])
                targets.extend([self.pad_token_id] * 2)
                j = i + 2
                for j in range(j, len(tokens)):
                    if self.ids_to_tokens[tokens[j]] != "<PROT_START>":
                        inputs.append(tokens[j])
                        targets.append(self.pad_token_id)
                    else:
                        i = j
                        break
                i = j

        
        assert len(inputs) == len(targets), "Mismatch between inputs and targets!"
        return inputs, targets

    

    """
    This tokenization scheme is for masking out ENTIRE proteins from the contig.
    Since the DNA kmers are kept, this is pushing the model to predict entire proteins
    from the DNA + AA context in a bidirectional way.
    """
    def mask_entire_proteins(self, tokens, mlm_probability=0.15):
        inputs = []
        targets = []

        prot_tag = self._convert_token_to_id("<PROT_START>")
        protein_start_idxs = [idx for idx in range(len(tokens)) if tokens[idx] == prot_tag]
        nu_proteins = len(protein_start_idxs)
        nu_to_mask = max(1, int(nu_proteins * mlm_probability))

        #print(tokens)
        #print(protein_start_idxs, nu_to_mask)

        selected_proteins_to_mask = random.sample(protein_start_idxs, nu_to_mask)
        
        i = 0
        while i < len(tokens):
            if i in selected_proteins_to_mask:
                inputs.extend([tokens[i], tokens[i+1]])
                targets.extend([self.pad_token_id] * 2)
                i = i + 2
                prot_end_idx = [j for j in range(i, len(tokens)) if tokens[j] == self._convert_token_to_id("<DNA_START>")]
                if len(prot_end_idx) == 0:
                    prot_end_idx = len(tokens)
                else:
                    prot_end_idx = prot_end_idx[0]
                inputs.extend([self.mask_token_id for _ in range(len(tokens[i:prot_end_idx]))])
                targets.extend(tokens[i:prot_end_idx])
                i = prot_end_idx
            else:
                inputs.extend([tokens[i]])
                targets.extend([self.pad_token_id])
                i += 1

        assert len(inputs) == len(targets), "Mismatch between inputs and targets!"
        return inputs, targets

    
    
    def _convert_token_to_type(self, token):
        return self.type_vocab.get(token)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def decode(self, tokens):
        return [self._convert_id_to_token(token) for token in tokens]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """
        Required for save_pretrained to work.
        """
        filename = "genomic_vocab.json"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
            
        path = os.path.join(save_directory, filename)
        with open(path, 'w') as f:
            json.dump(self.vocab, f)
            
        return (path,)

