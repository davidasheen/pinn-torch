import math

FIELD_NAMES = ("u", "v", "w", "p", "T", "Y", "c")
SPLITS = ("train", "valid", "test")
BC_SIDES = ("lo", "hi", "pair")
RESERVED_ARGS = ("X", "normal", "node", "segment")
PAIR_SUFFIXES = ("_lo", "_hi")
HISTORY_KEYS = ("epoch", "iteration", "phase", "train_total", "train_data", "train_pde", "train_bc",
                "val_total", "val_data", "val_pde", "val_bc")

SPLIT_PATTERN = (8, 1, 1)
SPLIT_ROUND_DECIMALS = 9
HIDDEN_LAYERS = (128, 128, 128, 128)
NUM_DOMAIN = 4000
NUM_BOUNDARY = 1000
RAR_POOL = 4.0
RAR_CHUNK = 32_000
RESAMPLE_EVERY = 200
RAR_BOUNDARY = True
BATCH_SIZE = None
ADAM_EPOCHS = 15_000
ADAM_LR = 1e-3
INVERSE_LR = 2e-2
GRAD_CLIP = 10.0
LBFGS_ITERATIONS = 5_000
LBFGS_CHUNK = 500
LBFGS_BOUND_TOL = 1e-2
LBFGS_TAIL_FRACTION = 0.5
LBFGS_DRIFT_SIGMA = 2.0

LBFGS_RESAMPLE = True
LBFGS_MAX_EVAL_FACTOR = 1.25
LBFGS_TOLERANCE_GRAD = 1e-8
LBFGS_HISTORY_SIZE = 100
LOG_EVERY = 500
RECORD_EVERY = 150
CHECKPOINT_EVERY = 500
SEED = 0
DETERMINISTIC = True
SEED_OFFSET_COLLOCATION = 1
SEED_OFFSET_BOUNDARY = 2
SEED_OFFSET_VALIDATION = 1000
SEED_OFFSET_BATCH = 2000
SEED_DRY_RUN = 12345
DRY_RUN_POINTS = 8
FEATURE_SHA_POINTS = 4096
TRAIN_DTYPE = "float32"
LBFGS_DTYPE = "float64"

PDE_WEIGHT = 1.0
RESUME_MODES = ("auto", True, False)
FEATURE_CHUNK = 2048
FINGERPRINT_KEYS = ("layers", "fields", "inverse", "known", "bounds", "pde", "boundary", "domain", "feature",
                    "split_pattern", "num_domain", "n_node", "rar_pool", "rar_chunk", "rar_boundary", "resample_every",
                    "batch_size", "adam_epochs", "adam_lr", "inverse_lr", "lbfgs_iterations", "lbfgs_chunk",
                    "lbfgs_resample", "seed", "seed_valid", "deterministic",
                    "sigma", "weights", "pde_weight", "dtype", "lbfgs_dtype", "data_sha")

COMSOL_VERSION = "6.3"
COMSOL_COMPONENT = "comp1"
COMSOL_MESH = "mesh1"
COMSOL_RECTANGLE_TYPE = "Rectangle"
COMSOL_SOLUTION_DATASET_TYPE = "Solution"
COMSOL_INITIAL_VALUE_TYPE = "init"
COMSOL_FLOW_PHYSICS_TYPES = ("LaminarFlow", "CreepingFlow", "TurbulentFlow")
COMSOL_SPECIES_PHYSICS_TYPES = ("DilutedSpecies",)
COMSOL_FLOW_FIELDS = ("p", "u", "v", "w")
COMSOL_SPECIES_FIELDS = ("c",)
COMSOL_SPATIAL_FIELDS = ("x", "y", "z")
COMSOL_FLOW_CONSTANTS = ("mu", "rho")
COMSOL_SPECIES_CONSTANTS = ("Dav_c",)
COMSOL_BC_ROLES = {"InletBoundary": "inlet", "OutletBoundary": "outlet", "WallBC": "wall",
                   "Symmetry": "symmetry", "Inflow": "inlet", "Outflow": "outlet",
                   "NoFlux": "no_flux", "Concentration": "concentration", "Flux": "flux"}
COMSOL_BC_EXTRA_PROPERTIES = ("BoundaryCondition", "MassFlowType", "BoundaryConditionType",
                              "FlowDirection", "SuppressBackflow", "Mn", "mnst", "stdmfr",
                              "p0", "c0", "U0in", "u0", "p_init", "u_init", "initc")
COMSOL_MATERIAL_PROPERTIES = ("density", "dynamicviscosity", "molarmass",
                              "heatcapacity", "ratioofspecificheat", "thermalconductivity")
COMSOL_CONSTANT_REL_SPAN = 1e-9
COMSOL_PYRAMID_TETS = ((0, 1, 3, 4), (0, 3, 2, 4))
COMSOL_PRISM_TETS = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
COMSOL_QUAD_TRIS = ((0, 1, 3), (0, 3, 2))

CFD_CASE = "base2_jun15"
CFD_PRECURSOR_MOLAR_MASS = 401.324
CFD_HALF_MODEL_FACTOR = 2.0
SCCM = 1e-6 / 60.0
CFD_QVOL_CHECK_TOL = 5e-3
CFD_MASS_BALANCE_TOL = 5e-3
CFD_TUBE_G_CUT = 72e-3
CFD_TUBE_AXIS_X = -38.1e-3
CFD_OUTLET_AXIS_X = 38.1e-3
CFD_TUBE_R_IN = 4.85e-3
CFD_TUBE_Z_BEND = -18.5e-3
CFD_MU_BOUNDS = (1e-6, 1e-4)
CFD_D_BOUNDS = (1e-6, 1e-3)
CFD_MU_INIT = 1e-5
CFD_D_INIT = 3e-5
CFD_LBFGS_ITERATIONS = 16_000

CFD_PULSE_LABEL = "Pulse 1//Solution 2"
CFD_T_WINDOW = (0.60, 2.40)
CFD_PULSE_N_DATA = 12_000
CFD_PULSE_NUM_BOUNDARY = {"symmetry": 1000, "wall_noslip": 1500, "wall_noflux": 1500, "bottom": 500,
                          "inlet": 300, "outlet": 300, "initial": 2000}
CFD_PULSE_FLOW_FIELDS = ("u", "v", "w", "p")

CFD_PURGE_LABEL = "Purge 1//Solution 3"
CFD_PURGE_N_DATA = 12_000
CFD_PURGE_NUM_BOUNDARY = {"symmetry": 1000, "wall_noflux": 1500, "bottom": 500, "inlet": 300, "outlet": 300,
                          "initial": 2000}
CFD_PURGE_SWEEP_CASES = ("purge_D", "purge2_D", "purge3_D", "purge4_D", "purge5_D",
                         "purge6_D", "purge7_D", "purge8_D", "purge9_D")
CFD_PULSE_FLOW_CASES = ("pulse_flow_mu", "pulse2_flow_mu", "pulse3_flow_mu", "pulse4_flow_mu", "pulse5_flow_mu",
                        "pulse6_flow_mu", "pulse7_flow_mu", "pulse8_flow_mu", "pulse9_flow_mu", "pulse10_flow_mu")
CFD_PULSE_CNET_CASES = ("pulse_c", "pulse2_c", "pulse3_c", "pulse4_c", "pulse5_c",
                        "pulse6_c", "pulse7_c", "pulse8_c", "pulse9_c", "pulse10_c")
CFD_SINGLE_CASES = CFD_PURGE_SWEEP_CASES + CFD_PULSE_FLOW_CASES + CFD_PULSE_CNET_CASES

CFD_MERGED_HOLDOUT = 3
CFD_MERGED_N_DATA = 12_000
CFD_MERGED_HIDDEN = (128,) * 6
CFD_MERGED_NUM_NODE = 24_000
CFD_MERGED_PURGE_SEEDS = (1, 2, 3, 4)
CFD_MERGED_PULSE_SEEDS = (1, 2)
CFD_MERGED_FLOW_CASES = (("pulse_all_flow_mu_sdisc",)
                         + tuple(f"pulse_all_flow_mu_sdisc_s{s}" for s in CFD_MERGED_PULSE_SEEDS))
CFD_MERGED_CNET_CASES = (("pulse_all_c_sdisc",)
                         + tuple(f"pulse_all_c_sdisc_s{s}" for s in CFD_MERGED_PULSE_SEEDS))
CFD_MERGED_FLOW_CASE = CFD_MERGED_FLOW_CASES[0]
CFD_MERGED_PURGE_CASES = (("purge_all_D_sdisc",)
                          + tuple(f"purge_all_D_sdisc_s{s}" for s in CFD_MERGED_PURGE_SEEDS))
CFD_MERGED_CASES = CFD_MERGED_FLOW_CASES + CFD_MERGED_CNET_CASES + CFD_MERGED_PURGE_CASES

CFD_CASES = CFD_SINGLE_CASES + CFD_MERGED_CASES

DIAG_BAND = 0.05
FIELD_UNITS = {"u": ("u [-]", 1.0), "v": ("v [-]", 1.0), "w": ("w [-]", 1.0),
               "p": ("p [-]", 1.0), "T": ("T [-]", 1.0), "Y": ("Y [-]", 1.0), "c": ("c [-]", 1.0)}
RESIDUAL_LABELS = {"continuity": "continuity", "momentum_u": "momentum u", "momentum_v": "momentum v",
                   "momentum_w": "momentum w", "energy": "energy", "species": "species"}
SPLIT_COLORS = {"train": "tab:blue", "valid": "tab:orange", "test": "tab:green"}

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
SWEEP_ADAM_EPOCHS = 6_000
SWEEP_LBFGS_ITERATIONS = 2_000
