from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------
# Dataset
# ———————————————————————————————————————————
_C.DATASET = CN()
_C.DATASET.PATH = '.\ACDC_data'

# -----------------------------------------------------------------------------
# Solver
# ---------------------------
# --------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.SEED = 2021
_C.SOLVER.LR = 0.000001
_C.SOLVER.BATCH_SIZE = 4
_C.SOLVER.NET = 'UNetpp'
# Could be 'UNet' or 'UNetpp' or 'ResUNetpp'
_C.SOLVER.N_CHANNELS = 1
_C.SOLVER.N_CLASSES = 4
_C.SOLVER.MAX_EPOCHS = 2000
_C.SOLVER.PATIENCE = 2000

_C.SOLVER.DEEPSUPERVISION = False  # Set True while need pruning in test


def get_cfg_defaults():
    return _C.clone()
