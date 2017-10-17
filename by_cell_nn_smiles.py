import os
import sys
import time
import numpy as np
import pandas as pd

from utils import ap_utils

seed = [0, 100, 2017]
np.random.seed(seed[0])

# TODO check why this one doesn't work
# Check for gpu and specify which to use
# ap_utils.choose_gpu(use_devices='single', verbose=True)

import matplotlib
matplotlib.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt

# file_path = os.path.dirname(os.path.realpath(__file__))  # dirname returns the directory name of pathname
# lib_path = os.path.abspath(os.path.join(file_path, '..', 'utils'))  # abspath returns the normalized absolutized version of the pathname
# sys.path.append(lib_path)

# Load data
root_dir = os.getcwd()
data_name = 'smiles'
data_file_name = 'BR:MCF7_smiles.csv'
data_file_path = os.path.join(root_dir, data_file_name)

assert os.path.exists(data_file_path), "The path {} was not found.".format(data_file_path)
data = pd.read_csv(data_file_path)

# from datasets import NCI60
# data = NCI60.load_by_cell_data('BR:MCF7', drug_features=['smiles'], subsample=None)

print('\nTotal SMILES strings: {}'.format(len(data['SMILES'])))
print('Total SMILES strings (unique): {}'.format(len(data['SMILES'].unique())))

smiles_len = data['SMILES'].apply(len)  # the vector smiles_len contains the length of smiles strings

# Remove smiles if they are too short/large
data = ap_utils.filter_on_smiles_len(data=data, smiles_len=smiles_len, thres_min=None, thres_max=None)

# Shuffle the dataset
if data.index.name:
    data = data.reset_index()
new_index = np.random.permutation(data.index)
data = data.reindex(index=new_index)
# or, for not a dataframe -->
#np.random.shuffle(data)
print('\n{}'.format(data.head()))

# Remove NSC column
# data = data.loc[:, data.columns != 'NSC']  # TODO: see if NSC affects/helps predictions

# Drop duplicates
# data = data.drop_duplicates()


'''
========================================================================================================================
Prepare the data (tokenize)
========================================================================================================================
'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

samples = [s for s in data['SMILES']]

# ===============================================
# # Canocalization: remove invalid SMILES based on CDK library
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

# tokenization_method = 'featurize_seq_smiles'
tokenization_method = 'featurize_3d_smiles'
X, tokenizer = ap_utils.tokenize_smiles(samples, tokenization_method=tokenization_method)

# Extract LCONC (another input feature in addition to SMILES sequences) and GROWTH (response var)
X_lconc = data['LCONC'].values.reshape(-1, 1)
Y = data['GROWTH'].values
print("Y = {} +- {}".format(np.mean(Y), np.std(Y)))

# Discretize the response variable y
y_even, thresholds, _ = ap_utils.discretize(y=Y, bins=5, cutoffs=None, min_count=0, verbose=False)

# Print shape
print('X shape {}'.format(X.shape))
print('X_lconc shape {}'.format(X_lconc.shape))
print('Y shape {}'.format(y_even.shape))


'''
========================================================================================================================
Train NN
========================================================================================================================
'''
t0 = time.time()
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

import keras
from keras import optimizers, losses, initializers

# Remove duplicates
# Since we're using only smiles strings, we want to remove all string dupicates
# _, idx = np.unique(X, axis=0, return_index=True)  # np.unique returns sorted unique values
# idx = np.sort(idx)  # by applying np.sort, we re-sort based on the original indexes
# x_train = X[idx, :]
# y_train = Y[idx]
# print('x_train shape {}'.format(x_train.shape))
# print('y_train shape {}'.format(y_train.shape))

# Embedding parameters
if tokenization_method == 'featurize_3d_smiles':
    embd_input_dim = None
    embd_output_dim = None
else:
    vocabsize = len(tokenizer.word_index)  # total number of unique characters in the dataset
    embd_input_dim = vocabsize + 1  # input dimnesion of the embedding layer
    embd_output_dim = 16  # output dimnesion of the embedding layer
    maxlen = smiles_len.max()  # length of input sequences (required if Flatten or Dense is used at some point after embedding)

# Metrics
#metrics = ['mae']
metrics = [ap_utils.r_square]

# Hyperparameters
initializer = initializers.glorot_uniform()
loss = losses.mae
epochs = 100
batch_size = 128
k_folds = 5

# Callbacks
callbacks = []
callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=True))

# Run k-fold CV
if k_folds == 1:
    skf = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=seed[0])
else:
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed[0])

hs = dict()
train_scores, val_scores = [], []
tests, preds = None, None
best_model = None
best_score = -np.Inf

for f, (train_idx, val_idx) in enumerate(skf.split(X, y_even)):
    print("\nFold {} out of {}".format(f + 1, k_folds))

    # TODO: plot histogram of the split (plot discretize data and original)
    # fig = plt.figure()
    # plt.hist(Y[train_idx], bins=20*len(thresholds), label='original')
    # #plt.hist(y_even[train_idx], bins=20, label='binned')
    # for ii in range(len(thresholds)):
    #     id = thresholds[ii-1] <= Y & Y < thresholds[ii-1]
    #     plt.hist(Y[id], bins=20, alpha=0.7)

    # Split the data
    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = Y[train_idx], Y[val_idx]
    x_train_lconc, x_lconc_val = X_lconc[train_idx], X_lconc[val_idx]

    # Create model
    #optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0)
    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = ap_utils.create_rnn_with_lconc(n_features=X.shape[-1], method='Conv1D', optimizer=optimizer,
                                           loss=loss, initializer=initializer, metrics=metrics,
                                           data_dim=len(X.shape),
                                           embd_input_dim=embd_input_dim, embd_output_dim=embd_output_dim)

    # Train model
    history = model.fit(x=[x_train, x_train_lconc], y=y_train,
                        validation_data=([x_val, x_lconc_val], y_val),
                        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    hs[f] = history.history
    # hs[f] = history

    # Store scores
    train_score = model.evaluate(x=[x_train, x_train_lconc], y=y_train, batch_size=batch_size, verbose=True)
    val_score = model.evaluate(x=[x_val, x_lconc_val], y=y_val, batch_size=batch_size, verbose=True)
    train_scores.append(train_score)
    val_scores.append(val_score)

    # print("  fold {}/{}: score = {:.3f}".format(f + 1, k_folds, val_score))
    # if val_score > best_score:
    #     best_model = model

print("\nRunning time: {:.2f} minutes.\n".format((time.time() - t0)/60))

print(model.summary())

# Plot loss vs epochs (averaged over k folds)
ap_utils.plot_learning_kfold(hs, history, savefig=True, img_name='holdoutfold_1dccc_mae_orgtoken_postpad_add_dense')

