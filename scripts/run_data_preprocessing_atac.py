import argparse
import sys
from pathlib import Path
import numpy as np
import os
import re
import pickle
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cellstory.preprocess import pretrain_dataset,gene_tokenizer
import scanpy as sc
from scipy.sparse import csr_matrix  
from tqdm import tqdm
def _parse_args():
    # argparse
    
    parser = argparse.ArgumentParser(
        description="The pre-training dataset is processed"
    )
    parser.add_argument(
        "--input_datasetlist",
        type=str,
        default="/t9k/mnt/code/CLM-access/dataset/datalist_ATAC_celltype.csv",
        help="for_finetune dataset list file",
    )
    parser.add_argument(
        "--cell_type_annotation",
        default=True
    )

    parser.add_argument(
        "--batch_label",
        default=False
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help="Directory to save data ",
    )

    parser.add_argument(
    "--ATAC_vocab_file",
    type=str,
    default="/t9k/mnt/code/CLM-access/dataset/ATAC_vocabulary_with_special_2000_unified.json",
    help="File containing the gene vocabulary, default to None. If None, will "
    "use the default gene vocabulary from scFormer, which use HGNC gene symbols.",
    )
   
    parser.add_argument(
    '--all_nonzero_value_set_1',
    type=int,
    default=True
    )

    parser.add_argument(
    '--context_length',
    type=int,
    default=2000
    )
    parser.add_argument(
    '--peak_length',
    type=int,
    default=600
    )

    parser.add_argument(
    '--context_select',
    default="random"  # random or truncation
    )
    
    parser.add_argument(
        "--append_cls",
        default=True
    )
    
    parser.add_argument(
        "--preprocessing",
        default=True
    )
    parser.add_argument(
        "--tokenizer",
        default=True
    )
    parser.add_argument(
        "--all_peaks",
        default=True
    )
   
 
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = _parse_args()


    if args.preprocessing:
         
        dataset_file_list, data_types = pretrain_dataset.read_csv_and_extract_columns(args.input_datasetlist)
        all_target_values = []
        end_sum = []
        for i in range(len(dataset_file_list)):
            
            adata = sc.read_h5ad(dataset_file_list[i],backed="r")
            if args.cell_type_annotation:
                
                celltype_id_labels = adata.obs["cell_type"].astype("category").cat.codes.values
                
                celltypes = adata.obs["cell_type"].unique()
                num_types = len(np.unique(celltype_id_labels))
                id2type = dict(enumerate(adata.obs["cell_type"].astype("category").cat.categories))
                with open(f'{args.output_dir}/id2type.pkl', 'wb') as f:
                    pickle.dump(id2type, f)
                
                    
            if args.batch_label:
                batch_id_labels = adata.obs["batch_id"].astype("category").cat.codes.values
                batchs = adata.obs["batch_id"].unique()
                num_types = len(np.unique(batch_id_labels))
                id2type = dict(enumerate(adata.obs["batch_id"].astype("category").cat.categories))
                with open(f'{args.output_dir}/id2type.pkl', 'wb') as f:
                    pickle.dump(id2type, f)


            gene_vocab = args.ATAC_vocab_file
                

            vocab = gene_tokenizer.GeneVocab.from_file(gene_vocab)
            # chr_columns = []
         
            
            def natural_sort_key(item):
                if ':' in item[0] and '-' in item[0]:
                    chromosome, positions = item[0].split(':')
                    chr_num = pretrain_dataset.chr_to_num(chromosome)
                    start, end = positions.split('-') 
            
                    chromosome_number = int(chr_num)
                    start_position = int(start)
                    return (chromosome_number, start_position)
                else:  

                    return float('inf'), float('inf')
     
            adata_var_index = adata.var.index
            sorted_adata_var_index = [i for i, _ in sorted(enumerate(adata_var_index), key=lambda x: natural_sort_key([x[1]]))]
           
            chr_start_end = [(pretrain_dataset.chr_to_num(s.split(':')[0]), int(s.split(':')[1].split('-')[0]), int(s.split(':')[1].split('-')[1])) for s in adata.var.index[sorted_adata_var_index]]  
        
 
                
            sorted_data = sorted(vocab.get_stoi().items(), key=natural_sort_key)
        
            indexes = [] 
            chroms = []
            starts = []  
            ends = []  
            
            for key, value in sorted_data:  
                if ':' in key and '-' in key:  
                    chromosome, positions = key.split(':')
                    chr_num = pretrain_dataset.chr_to_num(chromosome)
                    start, end = positions.split('-')  
                    chroms.append(chr_num)
                    indexes.append(value)  
                    starts.append(int(start))  
                    ends.append(int(end))

            patch_indices,region_counts = pretrain_dataset.map_points_to_regions_and_get_indices(chr_start_end ,chroms ,starts,ends,indexes)
            
            n_obs = adata.n_obs
            step =1000
            all_target_value=[]
            if i == 0:
                an = 0
            else:
                an =0
            for start in tqdm(range(an, n_obs, step)):
                end = start + step
                if end > n_obs:
                    end = n_obs
            # extract to memory

                ad_mem = adata[start:end].to_memory()[:, sorted_adata_var_index]
               
                # ad_mem.X[ad_mem.X > 0] = 1
                if not isinstance(ad_mem.X, csr_matrix):
                    ad_mem.X = csr_matrix(ad_mem.X)

                # ad_mem.X = ad_mem.X.astype(np.int32)
            
                
                target_values,gene_tokens,vocab = pretrain_dataset.load_anndata(ad_mem,data_types[i],args,patch_indices,vocab)
    
                if args.tokenizer:
                    patch_data,gene_ids= gene_tokenizer.tokenize_batch_edit(
                                                    data = target_values,
                                                    gene_ids=gene_tokens,
                                                    pad_token_id = vocab["<pad>"],
                                                    max_len=args.context_length,
                                                    target_length=args.peak_length,
                                                    pad_value = -2,
                                                    append_cls =  args.append_cls,
                                                    all_nonzero_value_set_1 = args.all_nonzero_value_set_1,
                                                    cls_id = vocab["<cls>"]
                                                )
                    if args.cell_type_annotation:
                        cell_type_label = celltype_id_labels[start:end, np.newaxis, np.newaxis] * np.ones((1, 1, patch_data.shape[2]))
                        patch_data = np.concatenate((patch_data, cell_type_label), axis=1)
                    if args.batch_label:
                        batch_label = batch_id_labels[start:end, np.newaxis, np.newaxis] * np.ones((1, 1, patch_data.shape[2]))
                        patch_data = np.concatenate((patch_data, batch_label), axis=1)
                    np.save(f'{args.output_dir}/dataset_{i}_{start}_{end}.npy', patch_data)  
                del ad_mem
            
          
            end_sum.append(end)
            gene_ids = np.array([vocab["<mask>"] if x is None else x for x in gene_ids])
                
            np.save(f'{args.output_dir}/gene_tokens.npy',gene_ids)
            vocab.save_json(args.output_dir + f"/vocab_{data_types[i]}.json") 
            adata.file.close()   
    
    
    parquet_files = [str(f) for f in Path(args.output_dir).glob("*.npy")]
    def sort_by_index(file_path):  
        match = re.search(r'dataset_(\d+)_(\d+)_(\d+)\.npy$', file_path)
        if match:  
            data_num,start, end =int(match.group(1)), int(match.group(2)), int(match.group(3))  
            return (data_num,start, end)
        else:  
            return float('inf'), float('inf')
    npy_files = sorted(parquet_files, key=sort_by_index) 
  
    npy_files=npy_files[:-1]


    max_ends = {} 
    for npy in   npy_files:
        match = re.search(r'dataset_(\d+)_(\d+)_(\d+)\.npy$', npy) 
        data_num,start, end =int(match.group(1)), int(match.group(2)), int(match.group(3))
       
        if data_num not in max_ends:  
            max_ends[data_num] = end  
        else:  
            max_ends[data_num] = max(max_ends[data_num], end) 
    total_sum = sum(max_ends.values())  
    print("npy_files",npy_files)
    arrays = []   
   

    if not os.path.exists(f"{args.output_dir}/large_data.bin"):  

        with open(f"{args.output_dir}/large_data.bin", 'wb') as bin_file:

            pass
    

    if args.cell_type_annotation or args.batch_label:
        mm = np.memmap(f"{args.output_dir}/large_data.bin", dtype=np.int8, mode='r+', shape = (total_sum, args.context_length+1,args.peak_length) )
    else:
        mm = np.memmap(f"{args.output_dir}/large_data.bin", dtype=np.int8, mode='r+', shape = (total_sum, args.context_length,args.peak_length) )

    
    for index,filename in tqdm(enumerate(npy_files)):
        match = re.search(r'dataset_(\d+)_(\d+)_(\d+)\.npy$', filename)  
        data_num,start, end =int(match.group(1)), int(match.group(2)), int(match.group(3))  

        array = np.load(filename, mmap_mode="r")  
        mm[start+sum(end_sum[0:data_num]):end+sum(end_sum[0:data_num])]=array

        
        print("finish:",index)
    print(mm[0])
    print(mm.shape)
    mm.flush()
  
    

             
   


    
    


    

  
  
 



