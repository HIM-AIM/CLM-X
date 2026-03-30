import logging
import sys
import os


# TODO make logger propagate to other modules
def init_logger(args):
    logger = logging.getLogger(args.exp_name)
    log_file = os.path.join(args.log_dir, "run.log")
    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.propagate = True
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # # file handler
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger
