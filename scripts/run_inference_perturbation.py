import os
import sys
import dotmap
import pickle
# set gpu number
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import lightning as pl
import scanpy as sc
from tqdm import tqdm
import json
# add code to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.beitv3_pl_value import BeitForPretrain as BeitForPretrain_Value
from cellstory.preprocess.input import (
    prepare_rna_inference_data

)

from cellstory.inference.inference_rna import rna_perturbation_metrics
from cellstory.utils import convert_to_path
from cellstory.logger import init_logger

# import experiment
from configs.config_eval import ex


def model_infer_rna(model, dataloader, args):
    adata = sc.read_h5ad(args.rna_h5ad)
    genes = {idx: name for idx, name in enumerate(adata.var["gene_name"].tolist())}

    rna_perturb = {}
    rna_real = {}
    for batch_data in tqdm(dataloader):
      
        # for inference using raw tokens without masking
        gene_id = batch_data["gene_id"].long().cuda()
        ctrl_value = batch_data["ctrl_value"].float().cuda()
        pert_flag = batch_data["pert_flag"].long().cuda()
        perturb_value = batch_data["perturb_value"].float() .cuda() 
        padding_mask = torch.zeros_like(gene_id).long().cuda()
      
        # reversed_rna_vocab = {v: k for k, v in genes.items()}
       
        pert_indices = torch.where(pert_flag == 1)[1]
       
    
        
        pert_idx=[]
        for idx in pert_indices:
            pert_idx.append(genes[idx.item()]+"+ctrl")
     
        with torch.no_grad():
            # -----error----- TODO: fix, using visual tokens
            outputs = model.beit3(
                atac_tokens=None,
                rna_tokens=gene_id,
                values_atac=None,
                values_rna=ctrl_value,
                atac_padding_position=None,
                rna_padding_position=padding_mask,
                attn_mask=None,
                pert_flag =pert_flag
            )
            # remove cls token
            rna_feats = outputs["encoder_out"]
            rna_features = model.mlm_scorer(rna_feats)

            for i in range(len(pert_idx)):
                if pert_idx[i] not in rna_perturb.keys():
                    rna_perturb[pert_idx[i]] =[]
                    rna_real[pert_idx[i]] =[]
                else:
                    rna_perturb[pert_idx[i]].append(rna_features[i].cpu())
                    rna_real[pert_idx[i]].append(perturb_value[i].cpu())
           
    for key in rna_perturb.keys():
        rna_perturb[key]=torch.stack(rna_perturb[key], dim=0).numpy()
        rna_real[key]=torch.stack(rna_real[key], dim=0).numpy()

    
    return rna_perturb,rna_real




def rna_inference(args):
    # init logger
    logger = init_logger(args)

    adata_obs, dataloader, rna_vocab_size, atac_vocab_size = prepare_rna_inference_data(
        args
    )

    # set vocab_size for RNA & ATAC
    # let rna_vocab_size=rna_vocab_size
    args.rna_vocab_size = rna_vocab_size
    # let atac_vocab_size=atac_vocab_size
    args.atac_vocab_size = atac_vocab_size
    logger.info(f"vocab size: RNA: {args.rna_vocab_size}, ATAC: {args.atac_vocab_size}")

    # load model checkpoint
    logger.info("loading the model parameters")
    model = BeitForPretrain_Value.load_from_checkpoint(
        args.model_load_path, map_location="cpu", config=args
    )
    model = model.cuda()
    model.eval()

   
    # inference from dataloader
    rna_perturb,rna_real = model_infer_rna(
        model, dataloader,args
    )



    # return adata
    return rna_perturb,rna_real




@ex.automain
def main(_config):
    # load config
    args_ = dotmap.DotMap(_config)
    # set repeatable seed
    pl.seed_everything(args_.seed)

    # init logger
    logger = init_logger(args_)

    # path settings
    args_.dirpath = convert_to_path(args_.dirpath)

    # create output directory if not exists
    if not os.path.exists(args_.dirpath):
        os.makedirs(args_.dirpath)
    # create log directory if not exists
    if not os.path.exists(args_.log_dir):
        os.makedirs(args_.log_dir, exist_ok=True)

    # Start task-specific logic
    if args_.task == "rnamlm":
        args_.rna_h5ad = convert_to_path(args_.rna_h5ad)
        

        logger.info("Start inference for RNA")
        rna_perturb,rna_real = rna_inference(args_)
        with open(os.path.join(args_.dirpath, 'rna_perturb.pkl'), 'wb') as f:
            pickle.dump(rna_perturb, f)
        with open(os.path.join(args_.dirpath, 'rna_real.pkl'), 'wb') as f:
            pickle.dump(rna_real, f)
        logger.info(f"预测结果已保存至 {os.path.join(args_.dirpath, 'rna_perturb.pkl')}")
        logger.info(f"真实值已保存至 {os.path.join(args_.dirpath, 'rna_real.pkl')}")
        logger.info("Finish inference for RNA")
        logger.info("Start calculating metrics for RNA")
        avg_mse,avg_pearson,avg_pearson_delta,avg_pearson_delta_de= rna_perturbation_metrics(args_,rna_perturb,rna_real)
        results = {
        "avg_de20_mse": float(avg_mse),
        "avg_de20_pearson": float(avg_pearson),
        "avg_pearson_delta": float(avg_pearson_delta), 
        "avg_pearson_delta_de": float(avg_pearson_delta_de)
       }
        print(results)
        with open(os.path.join(args_.dirpath, 'metrics_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"评估指标已保存至 {os.path.join(args_.dirpath, 'metrics_results.json')}")
        logger.info("Finish calculating metrics for RNA")
      


