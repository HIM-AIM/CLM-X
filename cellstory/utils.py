from pathlib import Path
from typing import Union

import scanpy as sc


def get_obs(h5ad: Union[str, Path]):
    """
    get obs from h5ad file
    """
    adata = sc.read_h5ad(h5ad, backed="r")
    adata_obs = adata.obs
    adata.file.close()
    return adata_obs


def convert_to_path(path):
    if isinstance(path, str):
        path = Path(path)
    return path
