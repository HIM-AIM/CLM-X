import os
from sacred import Experiment

ex = Experiment("CellStory")

CODE_REPO = os.path.dirname(os.path.dirname(__file__))


@ex.config
def config():

    seed = 42

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
    learning_rate = 1e-5

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
    every_n_train_steps = 200
    load_to_cpu = False

    # below params varies with the environment
    per_gpu_batchsize = 8 # you should define this manually with per_gpu_batch_size=#
    num_gpus = 2
    batch_size = 8
    grad_steps = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    pin_mem = True
    max_steps = 1000000 # 339420 // batch_size * max_epoch // num_gpus  # for one gpu, 3 epoch, need //num_gpus
    num_warmup_steps = 200 # max_steps * 0.1 // 1  # for one gpu,
    adam_weight_decay = 0.01  # the default setting
    end_lr = 0

################################################################################

################################################################################
@ex.named_config
def finetune_batch_correction():
    project_name = "CLM-X_finetune_batch_correction"
    exp_name = "batch_correction"
    task = "rnaatacmlm"
    model_task = "finetune"
    input_mod = "RNA + ATAC"
    max_epoch = 50
    learning_rate = 5e-5
    batch_correction = True
    use_batch_emb = False
    model_load_path = "../checkpoint/for_finetune/RNA+ATAC-pretrained.ckpt"
    dirpath = (
        "../save/batch_correction"
    )
    embedding_type = "cls"
    resume_from_checkpoint = None
    context_length = 2000  # 2000
    peak_length = 600  # 600
    log_dir = os.path.join(dirpath, "logs")

    atac_dataset_path = "../data/finetune/batch_correction/dataset1-4/atac_data"
    rna_dataset_path = "../data/finetune/batch_correction/dataset1-4/rna_data.dataset"
    atac_mask_ratio = 0
    rna_mask_ratio = 0
    pad_id = 1999  # 1999
    mask_id = 1998  # 1998
    atac_rna_cls = True
    val_ratio = 0.05
    features_dim = 2000
################################################################################

################################################################################
@ex.named_config
def finetune_translation():
    project_name = "CLM-X_finetune_translation"
    exp_name = "translation"
    task = "rnaatacmlm"
    model_task = "finetune"
    input_mod = "RNA + ATAC"
    max_epoch = 100
    learning_rate = 2e-5

    model_load_path = "../checkpoint/for_finetune/RNA+ATAC-pretrained.ckpt"
    dirpath = (
        "../save/translation"
    )
    output_dir = "../save/translation"
    resume_from_checkpoint = None

    log_dir = os.path.join(dirpath, "logs")

    base_data_path = "../data/finetune/translation/dataset4"

    atac_train_dataset_path = os.path.join(base_data_path, "atac_train")
    rna_train_dataset_path = os.path.join(base_data_path, "rna_train_processed.dataset")
    atac_test_dataset_path = os.path.join(base_data_path, "atac_test")
    rna_test_dataset_path = os.path.join(base_data_path, "rna_test_processed.dataset")

    context_length = 2000  # 2000
    peak_length = 600  # 600
    pad_id = 1999  # 1999
    mask_id = 1998  # 1998
    atac_mask_ratio = 1
    rna_mask_ratio = 1
    translation_to_atac = False
    translation_to_rna = True
    pred_full = False #True False
    rna_cls = False
    atac_cls = True
    rna_feats = False
    atac_feats = False
    val_ratio = 0.1
    features_dim = 2000 # dataset34-37: 36601 dataset39: 13431  dataset40-41: 36495 hvg:2000
################################################################################

################################################################################
@ex.named_config
def finetune_modality_fusion():
    project_name = "finetune_modality_fusion"
    exp_name = "finetune_modality_fusion"
    task = "rnaatacmlm"
    model_task = "finetune"
    input_mod = "RNA + ATAC"
    embedding_type = "cls"
    max_epoch = 50
    learning_rate = 1e-4
    model_load_path = "../checkpoint/for_finetune/RNA+ATAC-pretrained.ckpt"
    dirpath = (
        "../save/modality_fusion"
    )
    resume_from_checkpoint = None
    context_length = 2000  # 2000
    peak_length = 600  # 600
    pad_id = 1999  # 1999
    mask_id = 1998  # 1998
    log_dir = os.path.join(dirpath, "logs")

    atac_train_dataset_path = "../data/finetune/fusion/dataset5/batch1/atac_data"
    rna_train_dataset_path = "../data/finetune/fusion/dataset5/batch1/rna_batch1.dataset"
    embedding_modality = "mix" #  "mix"  "atac"  "rna"
    input_mod = "RNA + ATAC"
    atac_mask_ratio = 0
    rna_mask_ratio = 0
    modality_fusion = True
    val_ratio = 0.05
    features_dim = 2000


################################################################################

################################################################################
@ex.named_config
def finetune_cell_type_annotation():
    project_name = "CLM-X_cell_type_annotation"
    exp_name = "cell_type_annotation"
    task = "rnaatacmlm"
    model_task = "finetune"
    input_mod = "RNA + ATAC"
    max_epoch = 100
    learning_rate = 1e-5
    cell_type_annotation = True
    model_load_path = "../checkpoint/for_finetune/RNA+ATAC-pretrained.ckpt"
    dirpath = (
        "../save/cell_type_annotation"
    )
    output_dir = "../save/cell_type_annotation"
    resume_from_checkpoint = None
    context_length = 2000  # 2000  5000
    peak_length = 600   # 600   256
    pad_id = 1999   # 1999  4999
    mask_id = 1998  # 1998 4998

    log_dir = os.path.join(dirpath, "logs")

    base_data_path = "../data/finetune/cell_type_annotation/dataset4"

    rna_test_h5ad = os.path.join(base_data_path, "rna_test_normalized.h5ad")
    atac_train_dataset_path = os.path.join(base_data_path, "atac_train")
    rna_train_dataset_path = os.path.join(base_data_path, "rna_train.dataset")
    atac_test_dataset_path = os.path.join(base_data_path, "atac_test")
    rna_test_dataset_path = os.path.join(base_data_path, "rna_test.dataset")
    # atac_vocab_file = None
    embedding_modality = "mix" #  "mix"  "atac"  "rna"
    atac_mask_ratio = 0
    rna_mask_ratio = 0
    features_dim = 2000
    val_ratio = 0.1
    label_smoothing = 0.05
################################################################################

################################################################################
@ex.named_config
def finetune_perturbation():
    exp_name = "CLM-X_perturbation"
    task = "rnaatacmlm"
    model_task = "finetune"
    input_mod = "RNA"
    learning_rate = 1e-4
    perturbation = True
    embedding_type = "cls"
    mask_token = False
    context_length = 2000  # 2000  5000
    peak_length = 600   # 600   256
    pad_id = 1999   # 1999  4999
    mask_id = 1998  # 1998 4998
    features_dim = 2000
    tokenization = False
    model_load_path = "../checkpoint/for_finetune/RNA+ATAC-pretrained.ckpt"
    dirpath = (
        "../save/finetune_perturbation"
    )

    resume_from_checkpoint = None
    val_ration = 0
    log_dir = os.path.join(dirpath, "logs")

    rna_dataset_path = "../data/finetune/perturbation/replogle_k562_essential/train"

