# nammu


<p align="center">
  <img src="mascot.png" width="320" alt="Nammu">
</p>

## Usage
```python
from transformers import AutoModel, AutoTokenizer

# You can structure your sequence with strand tokens like this or provide protein or nucleotide sequences
# as inputs.
sequence_example_1 = "<+>MLKTLMPA<->acgtacgt"

model = AutoModel.from_pretrained("nazbijari/nammu", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("nazbijari/nammu", trust_remote_code=True)
```

## Recreate environment
```bash
conda env create -f environment.yml
conda activate nammu
pip install -r requirements.txt
```

If installing mamba-ssm or causal-conv-1d breaks: Installing a direct build wheel that is specific to your hardware and software specifications with pip is what we found to work.

# Description
Nammu is a mixed-modality genomic language model trained with bidirectional Mamba-1 blocks on TattaBio's OMG database. For details please see paper here: https://www.biorxiv.org/content/10.64898/2026.07.07.736993v1

## Models
| Model | Description | Link |
| --- | --- | --- |
| `nazbijari/nammu` | Nammu trained with a 20K context window on proteins first for 500K steps, contigs second for 250K steps. BiMamba used is adopted from Caduceus but adds an extra residual connection outside the BiMamba block. | https://huggingface.co/nazbijari/nammu |


## maglm
Code base containing model definitions for Nammu and other models compared against

## multimodal_benchmarking_scripts 
The multimodal_benchmarking_scripts folder contains the workflows used to perform benchmarking on function (deep sea metagenome KO function) and taxonomy (CAMI marine dataset). 

## results
Contains benchmarking results for all models in the paper, CAMI results, and code to run the KO benchmark task.

## train.py
All training logic, can be initiated in run.sh

## run.sh
Executes train.py. Provides and exampe of usage.


