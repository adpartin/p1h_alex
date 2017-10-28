from __future__ import print_function
import os
import sys
import time
import numpy as np
import pandas as pd
from argparser_nn import get_parser

from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

import keras
from keras import optimizers, losses, initializers

from utils import ap_utils


# Choose GPU
import matplotlib
matplotlib.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt  # this shuold be invoked after os.environ[]


# Initialize parser
description = 'Build neural network to process SMILES strings (sequence processing) for drug response prediction.'
parser = get_parser(description)
args = parser.parse_args()

print('Args:', args, end='\n\n')
print('Use percent growth for dose levels in log concentration range: [{}, {}]'.format(args.min_logconc, args.max_logconc))

seed = [0, 100, 2017]
np.random.seed(seed[0])


# Load data
# (data should be generated using dataframe.py)
root_dir = os.path.dirname(os.path.realpath(__file__))
# root_dir = os.getcwd()
data_file_name = args.file  # 'BR:MCF7_smiles.csv'
data_file_path = os.path.join(root_dir, data_file_name)

assert os.path.exists(data_file_path), "The following path was not found:  {}".format(data_file_path)
data = pd.read_csv(data_file_path)

# (another appraoch is to get data)
# from datasets import NCI60
# data = NCI60.load_by_cell_data('BR:MCF7', drug_features=['smiles'], subsample=None)

print('\nTotal SMILES strings: {}'.format(len(data['SMILES'])))
print('Total SMILES strings (unique): {}'.format(len(data['SMILES'].unique())))


# Remove SMILES based on length (if too short/large) --> add this to NCI60.py
smiles_len = data['SMILES'].apply(len)  # the vector smiles_len contains the length of smiles strings
data = ap_utils.filter_on_smiles_len(data=data, smiles_len=smiles_len, thres_min=None, thres_max=None)


# Drop duplicates
# data = data.drop_duplicates()

# Remove NSC column
# data = data.loc[:, data.columns != 'NSC']  # TODO: check if NSC helps prediction


'''
========================================================================================================================
Prepare the data (tokenize and split)
========================================================================================================================
'''
# Shuffle the dataset
if data.index.name:
    data = data.reset_index()
new_index = np.random.permutation(data.index)
data = data.reindex(index=new_index)
# or, if not a dataframe -->
#np.random.shuffle(data)
print('\n{}'.format(data.head()))

# SMILES to list of strings
samples = [s for s in data['SMILES']]

# ===============================================
# TODO: Canocalization: remove invalid SMILES based on CDK library
# from rdkit import Chem
# # smis = ["CN2C(=O)N(C)C(=O)C1=C2N=CN1C", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]
# # problematic: 813
# bad = [813, 1335, 1588, 3065, 3252, 3682, 3920, 4282, 4453, 4701, 5292, 5555, 5663, 5863, 6897, 7665, 7917, 8629, 8818, 9210]
# id = np.ones(len(data)).astype('bool')
# id[bad] = False
# data = data.iloc[id, :]
# samples = [s for i, s in enumerate(samples) if i not in bad]
# smis = []
# for i, smi in enumerate(samples):
#     #print(i, smi)
#     #if i not in bad:
#     smis.append(Chem.MolToSmiles(Chem.MolFromSmiles(smi), True))
# samples = smis
# ===============================================

# Choose tokenization method and tokenize
token_method = args.token  # 'seq_generic', 'seq_smiles', '3d_smiles'
X, tokenizer = ap_utils.tokenize_smiles(samples, token_method=token_method)

# Bound the values of GROWTH to [-1, 1]
data['GROWTH'] = data['GROWTH'].apply(lambda x: -1 if x < -1 else x)
data['GROWTH'] = data['GROWTH'].apply(lambda x: 1 if x > 1 else x)

# Extract LCONC (will be concatenated with SMILES as an input to NN) and GROWTH (response var)
X_lconc = data['LCONC'].values.reshape(-1, 1)
Y = data['GROWTH'].values

# Discretize the response variable y
y_even, thresholds, _ = ap_utils.discretize(y=Y, bins=5, cutoffs=None, min_count=0, verbose=False)

# Print shape
print('\nX shape {}'.format(X.shape))
print('X_lconc shape {}'.format(X_lconc.shape))
print('Y shape {}'.format(y_even.shape))


'''
========================================================================================================================
Train NN
========================================================================================================================
'''
t0 = time.time()
model_name = args.name  # 'model'

# Remove duplicates (if we use only SMILES w/o LCONC)
# _, idx = np.unique(X, axis=0, return_index=True)  # np.unique returns sorted unique values
# idx = np.sort(idx)  # by applying np.sort, we re-sort based on the original indexes
# x_train = X[idx, :]
# y_train = Y[idx]

# Input shape (involves embedding parameters)
maxlen = X.shape[1]  # smiles_len.max()
vocab_size = len(tokenizer.word_index) + 1
if token_method == '3d_smiles':
    embed_input_dim = None
    embed_output_dim = None
    input_shape = (maxlen, vocab_size,)
else:
    embed_input_dim = vocab_size  # input dimnesion of embedding layer
    embed_output_dim = 16  # output dimnesion of embedding layer
    input_shape = (maxlen,)

print('\nembd_input_dim: {}'.format(embed_input_dim))
print('embd_output_dim: {}'.format(embed_output_dim))
print('input_shape: {}'.format(input_shape))

# Metrics
metrics = [ap_utils.r_square]

# Hyperparameters
initializer = initializers.glorot_uniform(seed=seed[0])
loss = losses.mae
epochs = args.epochs  # 20
batch_size = args.batch  # 128
k_folds = args.cv  # 5
layer = args.layer  # 'conv1d'

# Callbacks
callbacks = []
callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=True))

# k-fold CV scheme
if k_folds == 1:
    skf = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=seed[0])
else:
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed[0])

# Run k-fold CV
hs = dict()
best_model = None
best_model_id = 0
best_score = -np.Inf
train_scores = pd.DataFrame(data=np.zeros((k_folds, len(metrics)+1)))
val_scores = pd.DataFrame(data=np.zeros((k_folds, len(metrics)+1)))

for f, (train_idx, val_idx) in enumerate(skf.split(X, y_even)):
    print("\nFold {} out of {}".format(f + 1, k_folds))

    # TODO: plot histogram of the split (plot discretize data and original)
    # fig = plt.figure()
    # plt.hist(Y[train_idx], bins=20*len(thresholds), label='original')
    # #plt.hist(y_even[train_idx], bins=20, label='binned')
    # for ii in range(len(thresholds)):
    #     id = thresholds[ii-1] <= Y & Y < thresholds[ii-1]
    #     plt.hist(Y[id], bins=20, alpha=0.7)

    # Split data
    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = Y[train_idx], Y[val_idx]
    x_train_lconc, x_val_lconc = X_lconc[train_idx], X_lconc[val_idx]

    # Get the optimizer
    if args.optimizer == 'adam':
        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif args.optimizer == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0)
    elif args.optimizer == 'sgd':
        optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # Create model
    model = ap_utils.create_rnn_with_lconc(input_shape=input_shape, layer=layer, optimizer=optimizer,
                                           loss=loss, initializer=initializer, metrics=metrics,
                                           data_dim=len(X.shape),
                                           embed_input_dim=embed_input_dim, embed_output_dim=embed_output_dim,
                                           model_fig_name=model_name)

    # Train model
    history = model.fit(x=[x_train, x_train_lconc], y=y_train,
                        validation_data=([x_val, x_val_lconc], y_val),
                        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    hs[f] = history

    # Store scores
    train_scores.iloc[f, :] = model.evaluate(x=[x_train, x_train_lconc], y=y_train, batch_size=batch_size, verbose=False)
    val_scores.iloc[f, :] = model.evaluate(x=[x_val, x_val_lconc], y=y_val, batch_size=batch_size, verbose=False)

    # Save best model based on loss
    if val_scores.iloc[f, 0] > best_score:
        best_score = val_scores.iloc[f, 0]
        best_model = model
        best_model_id = f

train_scores.columns = ap_utils.get_performance_metrics(history)
val_scores.columns = ap_utils.get_performance_metrics(history)


'''
========================================================================================================================
Summarize results
========================================================================================================================
'''
print(model.summary())

print("\nRunning time: {:.2f} minutes.".format((time.time() - t0)/60))

# Save scores into file
ap_utils.save_results(train_scores, val_scores, model_name)

# Print scores
ap_utils.print_results(train_scores, val_scores)

# Plot loss vs epochs (averaged over k folds)
ap_utils.plot_learning_kfold(hs, savefig=True, img_name=model_name + '_learn_kfold')
ap_utils.plot_learning(hs[best_model_id], savefig=True, img_name=model_name + '_learning_lr')

# Save model
model.save(model_name+'.h5')

