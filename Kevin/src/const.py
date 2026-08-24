import math

FIELD_NAMES = ("u", "v", "w", "p", "T", "Y")
SPLITS = ("train", "valid", "test")
BC_SIDES = ("lo", "hi", "pair")

SPLIT_PATTERN = (8, 1, 1)
SPLIT_ROUND_DECIMALS = 9

HIDDEN_LAYERS = (64, 64, 64, 64)
NUM_DOMAIN = 4000
NUM_BOUNDARY = 1000
RAR_POOL = 4.0
RAR_CHUNK = 32_000
RESAMPLE_EVERY = 200
ADAM_ITERATIONS = 15_000
ADAM_LR = 1e-3
GRAD_CLIP = 10.0
LBFGS_ITERATIONS = 5_000
LBFGS_MAX_EVAL_FACTOR = 1.25
LBFGS_TOLERANCE_GRAD = 1e-8
LBFGS_HISTORY_SIZE = 100
LOG_EVERY = 500
RECORD_EVERY = 150
CHECKPOINT_EVERY = 500
SEED = 0
SEED_OFFSET_COLLOCATION = 1
SEED_OFFSET_BOUNDARY = 2
TRAIN_DTYPE = "float32"
LBFGS_DTYPE = "float64"

TG_RHO = 1.0
TG_MU_TRUE = 0.05
TG_MU_INIT = 1.0
TG_N_SAMPLES = 3000
TG_X_RANGE = (0.0, 2.0 * math.pi)
TG_Y_RANGE = (0.0, 2.0 * math.pi)
TG_Z_RANGE = (0.0, 1.0)
TG_T_RANGE = (0.0, 1.0)
TG_GRID_N = 60
TG_CHECK_T_SNAPSHOTS = (0.0, 0.5, 1.0)
TG_CHECK_T_TOL = 0.05
TG_RECON_T = 0.5

SS_RHO = 1.0
SS_ALPHA = 0.12
SS_MU_TRUE = 0.08
SS_D_TRUE = 0.03
SS_DT_TRUE = 0.015
SS_U0 = 1.0
SS_TW = 1.0
SS_YW = 1.0
SS_MU_INIT = 1.0
SS_D_INIT = 1.0
SS_DT_INIT = 1.0
SS_N_SAMPLES = 3000
SS_X_RANGE = (0.0, 1.0)
SS_Y_RANGE = (0.0, 1.0)
SS_Z_RANGE = (0.0, 2.5)
SS_T_RANGE = (0.1, 1.0)
SS_Z_N = 200
SS_CHECK_T_SNAPSHOTS = (0.1, 0.4, 0.7, 1.0)
SS_CHECK_T_TOL = 0.03
SS_RECON_T = 0.7

NOISE_LEVELS = (0.0, 0.01, 0.05, 0.1)
SAMPLE_COUNTS = (3000, 1000, 300, 100, 30)
N_REPEATS = 3
SWEEP_ADAM_ITERATIONS = 6_000
SWEEP_LBFGS_ITERATIONS = 2_000

DIAG_BAND = 0.05
FIELD_UNITS = {"u": ("u [-]", 1.0), "v": ("v [-]", 1.0), "w": ("w [-]", 1.0),
               "p": ("p [-]", 1.0), "T": ("T [-]", 1.0), "Y": ("Y [-]", 1.0)}
RESIDUAL_LABELS = {"continuity": "continuity", "momentum_u": "momentum u", "momentum_v": "momentum v",
                   "momentum_w": "momentum w", "energy": "energy", "species": "species"}
SPLIT_COLORS = {"train": "tab:blue", "valid": "tab:orange", "test": "tab:green"}
