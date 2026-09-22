# nammu

## Use model through HuggingFace
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("nazbijari/nammu", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("nazbijari/nammu", trust_remote_code=True)
```

## Recreate environment
```bash
conda env create -f environment.yml
conda activate nammu
pip install -r requirements.txt
```

# Description
Nammu is a mixed-modality genomic language model trained with bidirectional Mamba-1 blocks on TattaBio's OMG database. For details please see paper here: https://www.biorxiv.org/content/10.64898/2026.07.07.736993v1

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


