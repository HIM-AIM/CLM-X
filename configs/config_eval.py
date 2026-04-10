import os
from sacred import Experiment
import datetime

ex = Experiment("Beit3")

CODE_REPO = os.path.dirname(os.path.dirname(__file__))


@ex.config
def config():
    project_name = "inference"
    seed = 1
    data_path = None

    ################################################################################
    # vocab setting
    # 1. dataset, 2. vocab_size, 3. vocab_file
    ################################################################################
    atac_vocab_size = None
    rna_vocab_size = None
    rna_vocab_file = (
        "../vocab/RNA.vocab.json"
    )
    atac_vocab_file = (
        "../vocab/ATAC.vocab.json"
    )

    ################################################################################
    # Transformer Setting
    ################################################################################
    encoder_layers = 12  # for debug
    encoder_embed_dim = 512

    encoder_attention_heads = 8
    encoder_ffn_embed_dim = 512

    # feedforward activation, relu, gelu, swish
    activation_fn = "gelu"
    activation_dropout = 0.0

    # architecture setting
    pre_norm = False
    multiway = True

    # -------------------- Optimizer Setting------------------
    optim_type = "adamw"
    learning_rate = 1e-4
    # attention dropout
    attention_dropout = 0.1
    # ffn layer dropout
    dropout = 0.1

    # ------------------ PL Trainer Setting------------------
    resume_from = None
    fast_dev_run = False
    val_check_interval = None
    test_only = False
    use_sharded_training = False
    resume_during_training = False
    checkpoint_activations = False

    # model save and load setting
    every_n_train_steps = 1_000
    load_to_cpu = False

    # below params varies with the environment
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 2
    batch_size = 8
    grad_steps = 4
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = "bf16-mixed"

    pin_mem = True

    max_steps = 100000 // num_gpus  # for one gpu, 3 epoch, need //num_gpus
    num_warmup_steps = 10000 // num_gpus  # for one gpu,
    adam_weight_decay = 0.01  # the default setting
    end_lr = 0


################################################################################
# RNA-ATAC inference config
################################################################################
@ex.named_config
def infer_rna_atac_fusion():
    exp_name = "RNA-ATAC-inference"
    task = "rnaatacmlm"
    learning_rate = 1e-4
    embedding_type = "cls"  # cls or avgpool
    dirpath = "../inference"
    output_dir = "../inference"

    log_dir = os.path.join(dirpath, "logs")
    model_load_path = "../save/fusion/"
    num_gpus = 2
    batch_size = 10
    # inference settings
    obsm_key = "cellstory_rna"
    raw_layer_key = "counts"
    rna_h5ad = "../data/finetune/fusion/dataset5/batch1/rna_batch1.h5ad"
    atac_dataset_path = "../data/finetune/fusion/dataset5/batch1/atac_data"
    rna_dataset_path = "../data/finetune/fusion/dataset5/batch1/rna_batch1.dataset"


    embedding_modality = "mix"
    model_task = "inference"
    input_mod = "RNA + ATAC"
    # tokenize settings

    context_length = 2000  # 2000  5000
    peak_length = 600    # 600   256
    pad_id = 1999   # 1999  4999
    atac_mask_ratio = 0
    rna_mask_ratio = 0
    mask_id = 1998  # 1998 4998
    num_classes = 15
    features_dim = 2000

################################################################################

################################################################################
@ex.named_config
def infer_rna_atac_batch():
    exp_name = "RNA-ATAC-inference"
    task = "rnaatacmlm"
    learning_rate = 1e-4
    embedding_type = "cls"  # cls or avgpool
    batch_correction = True
    dirpath = "../inference"
    output_dir = "../inference"
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = "../save/batch/"

    num_gpus = 1
    batch_size = 40
    # inference settings
    obsm_key = "cellstory_rna"
    raw_layer_key = "counts"

    rna_h5ad = "../data/finetune/batch_correction/dataset1-4/rna_data.h5ad"

    atac_dataset_path = "../data/finetune/batch_correction/dataset1-4/atac_data"
    rna_dataset_path = "../data/finetune/batch_correction/dataset1-4/rna_data.dataset"
    model_task = "inference"
    input_mod = "RNA + ATAC"
    atac_rna_cls = True
    rna_cls = False
    embedding_modality = "mix"

    context_length = 2000  # 2000  5000
    peak_length = 600  # 600   256
    pad_id = 1999  # 1999  4999
    atac_mask_ratio = 0
    rna_mask_ratio = 0
    mask_id = 1998  # 1998 4998
    num_classes = 15
    features_dim = 2000


################################################################################

################################################################################
@ex.named_config
def infer_rna_perturbation():
    exp_name = "RNA-inference"
    task = "rnamlm"
    learning_rate = 1e-4
    # mask prob: 0.4
    perturbation =True
    dirpath = "../inference"
    output_dir = "../inference"
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = "../save/perturbation/"
    num_gpus = 1
    batch_size = 12
    # inference settings

    rna_h5ad = "../data/finetune/perturbation/replogle_k562_essential/test_data.h5ad"
    train_h5ad = "../data/finetune/perturbation/replogle_k562_essential/train_data.h5ad"
    rna_dataset_path = "../data/finetune/perturbation/replogle_k562_essential/test"

    model_task = "inference"
    input_mod = "RNA"
    # RNA preprocess settings
    context_length = 2000  # 2000  5000
    peak_length = 600  # 600   256
    pad_id = 1999  # 1999  4999
    mask_id = 1998  # 1998 4998
    val_ration = 1