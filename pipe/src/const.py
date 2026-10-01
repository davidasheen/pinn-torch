SPLITS = ("train", "valid", "test")
BC_SIDES = ("lo", "hi")
HISTORY_KEYS = ("epoch", "phase", "train_total", "train_pde", "train_bc", "val_total", "val_pde", "val_bc")

SPLIT_PATTERN = (8, 1, 1)
HIDDEN_LAYERS = (128, 128, 128, 128)
NUM_DOMAIN = 4000
NUM_BOUNDARY = 1000
RAR_POOL = 4.0
RESAMPLE_EVERY = 200
ADAM_EPOCHS = 15_000
ADAM_LR = 1e-3
GRAD_CLIP = 10.0
LBFGS_ITERATIONS = 5_000
LBFGS_CHUNK = 500
LBFGS_HISTORY_SIZE = 100
LBFGS_MAX_EVAL_FACTOR = 1.25
LBFGS_TOLERANCE_GRAD = 1e-8
LOG_EVERY = 500
RECORD_EVERY = 150
CHECKPOINT_EVERY = 500
TRAIN_DTYPE = "float32"
LBFGS_DTYPE = "float64"
SEED = 0
SEED_OFFSET_COLLOCATION = 1
SEED_OFFSET_VALIDATION = 2
SEED_OFFSET_BOUNDARY = 3
RESUME_MODES = ("auto", True, False)
FINGERPRINT_KEYS = ("layers", "fields", "known", "bounds", "pde", "boundary", "domain", "split_pattern", "num_domain",
                    "n_node", "rar_pool", "resample_every", "adam_epochs", "adam_lr", "grad_clip", "lbfgs_iterations",
                    "lbfgs_chunk", "lbfgs_history", "seed", "dtype", "lbfgs_dtype", "transform", "loss_balance",
                    "log_input")
LOSS_BALANCE_METHODS = ("grad_norm",)

PISTON_T_MAX = 0.5
PISTON_NU = 1.0
PISTON_DELTA = 0.02
PISTON_T_POWER = 2.0
PISTON_QUAD_N = 32
PISTON_NUM_NODE = 6000
PISTON_N_PROJ = 800
PISTON_EVAL_R_N = 101
PISTON_EVAL_T_N = 201
PISTON_EVAL_T_LOG = 41
PISTON_SNAP_T = (0.0, 1e-3, 1e-2, 0.05, 0.1, 0.5)
N_REPEATS = 3

EXPLORE_HIDDEN = (32, 32, 32, 32)
GRAD_NORM_EVERY = 1000
GRAD_NORM_ALPHA = 0.9
LOG_T_EPS = PISTON_DELTA ** 2
