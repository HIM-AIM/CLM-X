
# CLM-X: A multimodal single-cell foundation model with flexible multi-way Transformer for unified scRNA-seq and scATAC-seq analysis

This repository is the official implementation of CLM-X: A multimodal single-cell foundation model with flexible multi-way Transformer for unified scRNA-seq and scATAC-seq analysis 



## Requirements

To install requirements:

```setup
conda create --name CLM-access  -c pyg -c pytorch -c nvidia -c xformers -c conda-forge -c bioconda 'python==3.10' 'pytorch-cuda==12.1' 'pytorch==2.1.2' torchtriton torchvision cudatoolkit xformers nccl py-opencv
conda activate CLM-access
conda install -c conda-forge deepspeed
conda install -c bioconda scvi-tools
conda install pandas numba scipy seaborn pyarrow scikit-learn poetry numpy
conda install -c conda-forge scanpy==1.10.2
pip install dotmap -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lightning -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchscale -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchtext -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sacred -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn==2.5.8 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scglue -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install snapatac2 -i https://pypi.tuna.tsinghua.edu.cn/simple


```


## Data preprocessing
Process the pre-training files and fine-tuning files in the paper.
```bash
python scripts/run_data_preprocessing_atac.py
python scripts/run_data_preprocessing_rna.py
```

## Pre-training

To train the model(s) in the paper, run this command:
Pre-training script: `scripts/run_pretrain.py`
Parameter configuration: `configs/config.py` （The path needs to be modified when in use）
`ATAC-RNA` Pre-training
```train
python run_pretrain.py with pretrain_atac_rna --force
```

## Fine-tuning

To fine-tuning batch effect correction in the paper, run this command:
Fine-tuning script: `scripts/run_finetune_batch_correction.py`
Parameter configuration: `configs/config_finetune.py` （The path needs to be modified when in use）
```bash
python run_finetune_batch_correction.py with finetune_batch_correction --force
```
To fine-tuning cell type annotation in the paper, run this command:
Fine-tuning script: `scripts/run_finetune_cell_type_annotation.py`
Parameter configuration: `configs/config_finetune.py` （The path needs to be modified when in use）
```bash
python run_finetune_cell_type_annotation.py with finetune_cell_type_annotation --force
```
To fine-tuning multimodal intergration in the paper, run this command:
Fine-tuning script: `scripts/run_finetune_fusion.py`
Parameter configuration: `configs/config_finetune.py` （The path needs to be modified when in use）
```bash
python run_finetune_fusion.py with finetune_modality_fusion --force
```
To fine-tuning translation in the paper, run this command:
Fine-tuning script: `scripts/run_finetune_translation.py`
Parameter configuration: `configs/config_finetune.py` （The path needs to be modified when in use）
```bash
python run_finetune_translation.py with finetune_translation --force
```
To fine-tuning perturbation in the paper, run this command:
Fine-tuning script: `scripts/run_finetune_perturbation.py`
Parameter configuration: `configs/config_finetune.py` （The path needs to be modified when in use）
```bash
python run_finetune_perturbation.py with finetune_perturbation --force
```

## Evaluation

Parameter configuration: `configs/config_eval.py` (modify paths before running)

`run_inference_embedding.py`: export embeddings for batch effect correction and multimodal integration evaluation
`run_inference_perturbation.py`: perturbation task evaluation


Embedding export (for batch effect correction / multimodal integration)
```bash
python run_inference_embedding.py with infer_rna_atac_batch --force / infer_rna_atac_fusion --force
```
Perturbation evaluation
```bash
python run_inference_perturbation.py with infer_rna_perturbation --force
```

## Data and checkpoints
the data and checkpoints for Pre-training and Fine-tuning can be downloaded from the following link: `https://zenodo.org/records/19334485`




















