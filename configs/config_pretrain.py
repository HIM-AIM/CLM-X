import os
from sacred import Experiment

ex = Experiment("CellStory")

CODE_REPO = os.path.dirname(os.path.dirname(__file__))


@ex.config
def config():
    project_name = "cellcogni-for_finetune-mix"
    seed = 1

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

    encoder_layers = 12
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
    every_n_train_steps = 4000
    load_to_cpu = False

    # below params varies with the environment
    per_gpu_batchsize = 12 # you should define this manually with per_gpu_batch_size=#
    num_gpus = 4
    batch_size = 12
    grad_steps = 1
    num_nodes = 1
    load_path = ""
    num_workers = 16
    precision = '16-mixed'

    pin_mem = True
    max_epoch = 6
    max_steps = 2000000 # 339420 // batch_size * max_epoch // num_gpus  # for one gpu, 3 epoch, need //num_gpus
    num_warmup_steps = 10000 # max_steps * 0.1 // 1  # for one gpu,
    adam_weight_decay = 0.01  # the default setting
    end_lr = 0


################################################################################
# RNA pretraining with cls config
################################################################################
@ex.named_config
def pretrain_rna():
    exp_name = "RNA-for_finetune-cls"
    task = "rnamlm"
    learning_rate = 1e-4
    # mask prob: 0.4

    dirpath = (
        "../checkpoints/your_path/"
    )
    log_dir = os.path.join(dirpath, "logs")

    # preprocess & tokenize input settings
    rna_dataset_path = "../dataset"

    input_mod = "RNA"


################################################################################
# ATAC pretraining with cls config
################################################################################
@ex.named_config
def pretrain_atac():
    exp_name = "ATAC-for_finetune-cls"
    task = "atacmlm"

    learning_rate = 1e-4

    batch_size = 20
    num_gpus = 1

    dirpath = (
        "../checkpoints/your_path/"
    )
    log_dir = os.path.join(dirpath, "logs")

    # preprocess & tokenize input settings
    atac_dataset_path = "../dataset"

    input_mod = "ATAC"

################################################################################
# RNA + ATAC pretraining config
################################################################################
@ex.named_config
def pretrain_atac_rna():
    exp_name = "RNA+ATAC-for_finetune"
    task = "rnaatacmlm"
    # data_path = None
    learning_rate = 1e-5
    pretrain = True
    drop_path_rate = 0.1

    dirpath = (
        "../checkpoints/your_path/"
    )
    log_dir = os.path.join(dirpath, "logs")
    resume_from_checkpoint = None
    model_load_path_rna = "../checkpoint/for_multimodal_pretrain/RNA-pretrained.ckpt"
    model_load_path_atac = "../checkpoint/for_multimodal_pretrain/ATAC-pretrained.ckpt"


    atac_dataset_path = "../dataset"
    rna_dataset_path = "../dataset"
    model_task = "for_finetune"
    input_mod = "RNA + ATAC"
    tokenization_style = "rna"
    phase_change_epoch = 30

    both_pretrain = True

