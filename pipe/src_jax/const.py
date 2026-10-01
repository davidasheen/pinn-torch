import math

SPLITS = ("train", "valid", "test")
BC_SIDES = ("lo", "hi", "pair")
PAIR_SUFFIXES = ("_lo", "_hi")
HISTORY_KEYS = ("epoch", "phase", "train_total", "train_data", "train_pde", "train_bc",
                "val_total", "val_data", "val_pde", "val_bc")

SPLIT_PATTERN = (8, 1, 1)
HIDDEN_LAYERS = (128, 128, 128, 128)
NUM_DOMAIN = 4000
NUM_BOUNDARY = 1000
RAR_POOL = 4.0
RESAMPLE_EVERY = 200
ADAM_EPOCHS = 15_000
ADAM_LR = 1e-3
INVERSE_LR = 2e-2
GRAD_CLIP = 10.0
LBFGS_ITERATIONS = 5_000
LBFGS_CHUNK = 500
LBFGS_HISTORY_SIZE = 100
LOG_EVERY = 500
RECORD_EVERY = 150
CHECKPOINT_EVERY = 500
TRAIN_DTYPE = "float32"
LBFGS_DTYPE = "float64"
SEED = 0
SEED_OFFSET_NETWORK = 0
SEED_OFFSET_COLLOCATION = 1
SEED_OFFSET_BOUNDARY = 2
SEED_OFFSET_VALIDATION = 1000
RESUME_MODES = ("auto", True, False)
FINGERPRINT_KEYS = ("layers", "fields", "inverse", "known", "bounds", "pde", "boundary", "domain",
                    "split_pattern", "num_domain", "n_node", "rar_pool", "resample_every",
                    "adam_epochs", "adam_lr", "inverse_lr", "grad_clip", "lbfgs_iterations", "lbfgs_chunk",
                    "lbfgs_history", "seed", "dtype", "lbfgs_dtype", "data_sha")
FINGERPRINT_OPTIONAL_KEYS = ("transform", "loss_balance", "subnets", "log_input")
LOSS_BALANCE_METHODS = ("grad_norm",)

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
TG_P_REF_XY = (math.pi / 4, math.pi / 4)

REL_L2_TOL = 0.01
EVAL_Z = (0.0, 0.5, 1.0)
EVAL_T_N = 11

PISTON_T_MAX = 0.5
PISTON_NU = 1.0
PISTON_DELTA = 0.02
PISTON_T_POWER = 2.0
PISTON_QUAD_N = 32
PISTON_NUM_NODE = 6000
PISTON_N_TERMS = 2000
PISTON_N_PROJ = 800
PISTON_EVAL_R_N = 101
PISTON_EVAL_T_N = 201
PISTON_EVAL_T_LOG = 41
PISTON_T_SEGMENTS = (0.0, 0.01, 0.1, 0.5)
PISTON_SNAP_T = (0.0, 1e-3, 1e-2, 0.05, 0.1, 0.5)
PISTON_FIG9_T = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
PISTON_FIG12_T0 = 0.01
PISTON_FIG12_T = ((1e-4, 5e-4, 1e-3, 5e-3, 1e-2), (0.01, 0.05, 0.1, 0.5))
PISTON_SWEEP_WIDTHS = (16, 32, 64, 128, 256)
PISTON_SWEEP_DEPTHS = (2, 4, 8, 16)
PISTON_CHANNEL_HIDDEN = (32, 32, 32, 32)
SPARROW_CHANNEL_ALPHA = (4.49341, 7.72525, 10.9041, 14.0662, 17.2207)

EXPLORE_HIDDEN = (32, 32, 32, 32)
GRAD_NORM_EVERY = 1000
GRAD_NORM_ALPHA = 0.9
GRAD_NORM_CAPS = (100.0, 1000.0)
LOG_T_EPS = PISTON_DELTA ** 2

NOISE_LEVELS = (0.0, 0.01, 0.05, 0.1)
SAMPLE_COUNTS = (3000, 1000, 300, 100, 30)
N_REPEATS = 3
SWEEP_ADAM_EPOCHS = 6_000
SWEEP_LBFGS_ITERATIONS = 2_000

DIAG_BAND = 0.05
FIELD_UNITS = {"u": ("u [-]", 1.0), "v": ("v [-]", 1.0), "w": ("w [-]", 1.0), "p": ("p [-]", 1.0)}
RESIDUAL_LABELS = {"continuity": "continuity", "momentum_u": "momentum u", "momentum_v": "momentum v",
                   "momentum_w": "momentum w"}
SPLIT_COLORS = {"train": "tab:blue", "valid": "tab:orange", "test": "tab:green"}
