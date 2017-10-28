import argparse

# (ap) ----------------------------
# === Data pre-proc ===
THRES_FRAC_ROWS = 1
THRES_FRAC_COLS = 1
THRES_VAR = 0
THRES_CORR = 1
THRES_DISCRETE = 0
ONEHOT_DISCRETE = 'no'
CREATE_IOM = 'no'

MIN_GROWTH_BOUND = -1.0
MAX_GROWTH_BOUND = 1.0
# (ap) ----------------------------

CELLS = ['BR:MCF7']
DRUGS = ['1']
CELL_FEATURES = ['expression']
DRUG_FEATURES = ['descriptors']
MODELS = ['XGBoost', 'RandomForest']
CUTOFFS = [-0.50, 0.25]
CV = 5
FEATURE_SUBSAMPLE = 0
LOGCONC = -4.0
MIN_LOGCONC = -5.0
MAX_LOGCONC = -4.0
SCALING = 'std'
SUBSAMPLE = None
THREADS = -1
OUT_DIR = '.'


def get_parser(description=None):
    parser = argparse.ArgumentParser(description=description)

    # (ap) -------------------------------------------------------------------------------------------------------------
    parser.add_argument("--thres_frac_rows", type=float, default=THRES_FRAC_ROWS,
                        help="drop rows in which the total number of missing values exceeds this threshold: [0, 1]")

    parser.add_argument("--thres_frac_cols", type=float, default=THRES_FRAC_COLS,
                        help="drop cols in which the total number of missing values exceeds this threshold: [0, 1]")

    parser.add_argument("--thres_var", type=float, default=THRES_VAR,
                        help="drop cols with variance lower than this threshold: [0, 1]")

    parser.add_argument("--thres_corr", type=float, default=THRES_CORR,
                        help="drop cols whose correlations with other cols exceeds this threshold: [0, 1]")

    parser.add_argument("--thres_discrete", type=int, default=THRES_DISCRETE,
                        help="max number of unique values in a col (excluding na values) to consider the col as discrete")

    parser.add_argument("--onehot_discrete", default=ONEHOT_DISCRETE, metavar=ONEHOT_DISCRETE,
                        choices=['yes', 'no'],
                        help="whether to onehot encode the discrete features: 'no' or 'yes'")

    parser.add_argument("--create_iom", default=CREATE_IOM, metavar=CREATE_IOM,
                        choices=['yes', 'no'],
                        help="whether to create an indicator of missingness (add binary columns indicating missing"
                             "values): 'no' or 'yes'")

    parser.add_argument("--min_growth_bound", type=float, default=MIN_GROWTH_BOUND,
                        help="bound (cap) all growth values to this low value")

    parser.add_argument("--max_growth_bound", type=float, default=MAX_GROWTH_BOUND,
                        help="bound (cap) all growth values to this max value")
    # (ap) -------------------------------------------------------------------------------------------------------------

    parser.add_argument("-c", "--cell_features", nargs='+', default=CELL_FEATURES, metavar='CELL_FEATURES',
                        choices=['expression', 'mirna', 'proteome', 'all', 'expression_5platform'],
                        help="use one or more cell line feature sets: 'expression', 'mirna', 'proteome', 'all'; "
                             "use all for ['expression', 'mirna', 'proteome']")

    parser.add_argument("-d", "--drug_features", nargs='+', default=DRUG_FEATURES, metavar='DRUG_FEATURES',
                        choices=['descriptors', 'latent', 'all', 'noise', 'smiles'],  # (ap) added smiles
                        help="use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, "
                             "or both, SMILES strigns, or random features; 'descriptors', 'latent', 'all', 'smiles', "
                             "'noise'")  # (ap) added smiles

    parser.add_argument("-m", "--models", nargs='+', default=MODELS,
                        help="list of regression models: XGBoost, XGB.1K, XGB.10K, RandomForest, RF.1K, RF.10K, AdaBoost, "
                             "Linear, ElasticNet, Lasso, Ridge; or list of classification models: XGBoost, XGB.1K, "
                             "XGB.10K, RandomForest, RF.1K, RF.10K, AdaBoost, Logistic, Gaussian, Bayes, KNN, SVM")

    parser.add_argument("--cells", nargs='+', default=CELLS,
                        help="list of cell line names")

    parser.add_argument("--drugs", nargs='+', default=DRUGS,
                        help="list of drug NSC IDs")

    parser.add_argument("--cv", type=int, default=CV,
                        help="cross validation folds")

    parser.add_argument("--classify",  action="store_true",
                        help="convert the regression problem into classification based on category cutoffs")

    parser.add_argument("--autobins", type=int, default=0,
                        help="number of evenly distributed bins to make when classification mode is turned on")

    parser.add_argument("--cutoffs", nargs='+', type=float, default=CUTOFFS,
                        help="list of growth cutoffs (between -1 and +1) delineating response categories")

    parser.add_argument("--feature_subsample", type=int, default=FEATURE_SUBSAMPLE,
                        help="number of features to randomly sample from each category, 0 means using all features")

    parser.add_argument("--logconc", type=float, default=LOGCONC,
                        help="log concentration of dose response data to use: -3.0 to -7.0")

    parser.add_argument("--min_logconc", type=float, default=MIN_LOGCONC,
                        help="min log concentration of dose response data to use: -3.0 to -7.0")

    parser.add_argument("--max_logconc",  type=float, default=MAX_LOGCONC,
                        help="max log concentration of dose response data to use: -3.0 to -7.0")

    parser.add_argument("--use_gi50",  action="store_true",
                        help="use NCI GI50 value instead of percent growth at log concentration levels")

    parser.add_argument("--scaling", default=SCALING, metavar='SCALING',
                        choices=['minabs', 'minmax', 'std', 'none'],
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit"
                             "normalization; 'none': no normalization")

    parser.add_argument("--subsample", default=SUBSAMPLE, metavar='SUBSAMPLE',
                        choices=['naive_balancing', 'none'],
                        help="dose response subsample strategy; 'none' or 'naive_balancing'")

    parser.add_argument("--threads", type=int, default=THREADS,
                        help="number of threads per machine learning training job; -1 for using all threads")

    parser.add_argument("-o", "--out_dir", default=OUT_DIR,
                        help="output directory")

    return parser

