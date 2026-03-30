# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import copy

import torch.nn as nn


def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module

def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.atac = module
        self.rna = copy.deepcopy(module)
        self.rna.reset_parameters()
        self.armix = copy.deepcopy(module)
        self.armix.reset_parameters()

        self.split_position = None

    def forward(self, x, **kwargs):
        if self.split_position is None:
            raise NotImplementedError("self.split_position。")

        # 1) 当 split_position == -1 => 'atac'
        if self.split_position == -1:
            return self.atac(x, **kwargs)

        # 2) 当 split_position == 0 => 'rna'
        if self.split_position == 0:
            return self.rna(x, **kwargs)

        return self.armix(x, **kwargs)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.atac = modules[0]
        self.rna = modules[1]
        self.split_position = -1