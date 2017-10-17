""" Created by Alex Partin """
import os
import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras import Input
from keras import layers, optimizers, losses
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def choose_gpu(use_devices='single', verbose=True):
    """ Check if any gpu's are visible and decide how many to use.
    An alternative (better?) approach to this function can be implemented in bash script that also checks if GPU's are
    not just visible but also not available (parse the output of 'nvidia-smi').

    http://www.acceleware.com/blog/cudavisibledevices-masking-gpus

    Args:
        use_devices (str): specifuy which gpu devices to use.
            'single': use a single device (default is 0)
            'all': use all devices
            '0, 2': use device 0 and 2
    """
    # TODO for some reason it activates all the visible gpu's! take care of this!

    # Check all visible devices
    from tensorflow.python.client import device_lib
    device_names = [device.name for device in device_lib.list_local_devices()]
    print("\nVisibile devices: {}".format(device_names))

    # Check how many gpu's are visible
    n_gpu = np.asarray([True if 'gpu' in n else False for n in device_names]).sum()
    print("Visibile GPU's (total): {}".format(n_gpu))

    # Return if no gpu are visible
    if n_gpu == 0 and verbose:
        print("No GPU's found.")
        return
    elif n_gpu == 0:
        return

    # Get GPU names
    gpu = [n for n in device_names if 'gpu' in n]

    # Set enviroment variable CUDA_VISIBLE_DEVICES to specify which gpu to use
    if use_devices == 'single':
        print("use_devices='single'")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif use_devices == 'all':
        print("use_devices='all'")
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([device[device.find(':')+1:] for device in gpu])
    else:
        print("use_devices='else'")
        # TODO add condition/exception that checks that legal use_devices provided
        os.environ["CUDA_VISIBLE_DEVICES"] = use_devices

    # This is required when we execute plotting on remote machine!
    # Force matplotlib to not use any Xwindows backend
    # Allows to display figures on remotes and must come before import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    if verbose:
        print("\nUsing GPU devices (device index):  {}.\n".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    return


def r_square(y_true, y_pred):
    """ R^2 learning `metric` for model.fit(). """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r_sqr = (1.0 - SS_res/(SS_tot + K.epsilon()))
    return r_sqr


def r_square_adjusted(y_true, y_pred, n, k):
    """ Radj^2 learning `metric` for model.fit().
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2

    Args:
        n: number of tranining samples (sample size)
        k: number of features (exploratory/independent variables)
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r_sqr = (1.0 - SS_res/(SS_tot + K.epsilon()))
    r_sqr_adj = 1 - (1 - r_sqr) * ((n - 1)/(n - k - 1))
    return r_sqr_adj


def plot_learning_kfold(hs, history, savefig=True, img_name=None):
    """ Plot learning progress (averaged across k folds).

    Args:
        hs:
        history: a callback object from keras model tranining (model.fit)
        metrics:
        savefig:
        img_name:
    """
    # TODO eliminate the need to pass history
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [10, 8]
    legend_font_size = 10

    epochs = np.asarray(history.epoch) + 1
    k_folds = len(hs)

    # Extract names of all recorded performance metrics for training and val sets
    all_metrics = list(hs[0].keys())  # all metrics including everything returned from callbacks
    pr_metrics = []  # performance metrics recorded for train and val such as 'loss', etc. (excluding callbacks)
    for m in all_metrics:
        if 'val' in m:
            pr_metrics.append('_'.join(m.split('_')[1:]))

    # Plot
    for m in pr_metrics:
        metric_name = m
        metric_val_name = 'val_' + m

        # Compute avg across folds
        metric_avg = np.asarray([hs[fold][metric_name] for fold in hs]).sum(axis=0, keepdims=True) / k_folds
        metric_val_avg = np.asarray([hs[fold][metric_val_name] for fold in hs]).sum(axis=0, keepdims=True) / k_folds

        # Plot metrics for each fold vs epochs
        # TODO the code for plotting below can be combined with plot_learning()
        marker = ['b.', 'r^', 'kx', 'mv', 'gp', 'bs', 'r8', 'kD']
        fig = plt.figure()
        for i, metric in enumerate([metric_name, metric_val_name]):
            ax = fig.add_subplot(3, 1, i + 1)
            for fold in range(k_folds):
                plt.plot(epochs, hs[fold][metric], label="fold{}".format(fold + 1))
            plt.ylabel(metric)
            plt.grid('on')
            plt.xlim([0.5, len(epochs) + 0.5])
            plt.ylim([0, 1])
            plt.legend(loc='best', prop={'size': legend_font_size})
        # plt.savefig('kfold_cv')

        # Plot the average of metrics across folds vs epochs
        ax = fig.add_subplot(3, 1, 3)
        plt.plot(epochs, metric_avg.flatten(), 'bo', label=metric_name)
        plt.plot(epochs, metric_val_avg.flatten(), 'rx', label=metric_val_name)
        plt.ylabel('loss avg over folds')
        plt.xlabel('Epochs')
        plt.grid('on')
        plt.xlim([0.5, len(epochs) + 0.5])
        plt.ylim([0, 1])
        plt.legend(loc='best', prop={'size': legend_font_size})

        if savefig:
            if img_name:
                plt.savefig(img_name + '_' + metric_name)
            else:
                plt.savefig('learning_kfold')


# # =====================
#     loss_avg = np.asarray([hs[f]['loss'] for f in hs]).sum(axis=0, keepdims=True) / len(hs)
#     val_loss_avg = np.asarray([hs[f]['val_loss'] for f in hs]).sum(axis=0, keepdims=True) / len(hs)
#     # train_loss = np.asarray([hs[f]['loss'] for f in range(len(hs))]).mean(axis=0)
#     # train_metric = np.asarray([hs[f]['r_square'] for f in range(len(hs))]).mean(axis=0)
#
#     # Plot metrics for each fold vs epochs
#     marker = ['bo', 'b^', 'bx', 'bv', 'bp', 'bs', 'b8', 'bD']
#     fig = plt.figure()
#     for i, metric in enumerate(['loss', 'val_loss']):
#         ax = fig.add_subplot(3, 1, i + 1)
#         for fold in range(len(hs)):
#             plt.plot(epochs, hs[fold][metric], label="fold{}".format(fold + 1))
#         plt.ylabel(metric)
#         plt.grid('on')
#         plt.xlim([0.5, len(epochs) + 0.5])
#         plt.ylim([0, 1])
#         plt.legend(loc='best', prop={'size': legend_font_size})
#     # plt.savefig('kfold_cv')
#
#     # Plot the average of metrics across folds vs epochs
#     ax = fig.add_subplot(3, 1, 3)
#     plt.plot(epochs, loss_avg.flatten(), 'bo', label='loss')
#     plt.plot(epochs, val_loss_avg.flatten(), 'rx', label='val_loss')
#     plt.ylabel('loss avg over folds')
#     plt.xlabel('Epochs')
#     plt.grid('on')
#     plt.xlim([0.5, len(epochs) + 0.5])
#     plt.ylim([0, 1])
#     plt.legend(loc='best', prop={'size': legend_font_size})
#
#     if savefig:
#         if img_name:
#             plt.savefig(img_name)
#         else:
#             plt.savefig('learning_kfold')
# # =====================


def plot_learning(history, metrics=None, savefig=True, img_name=None):
    """ Plot learning progress for all recorded metrics. This function should be used with hold-out validation scheme
    since it allows to plot learning rate on a separate axis.

    Args:
        history: a callback object from keras model tranining (model.fit)
        metrics:
        savefig:
        img_name:
    """
    legend_font_size = 10
    import matplotlib.pyplot as plt
    hs = history.history.copy()
    epochs = np.asarray(history.epoch) + 1

    # TODO find all keys with 'val' substring and the corresponding equivalent metrics of the training set
    if 'lr' in hs:
        lr = hs['lr']
        hs.pop('lr', None)  # temporary solution

    metrics = list(hs.keys())  # list of all the recorded metrics during training

    m = len(metrics) // 2  # number of unique metrics
    fig = plt.figure()
    for p in range(m):
        ylabel_name = '_'.join(metrics[p].split('_')[1:])
        ax = fig.add_subplot(m, 1, p + 1)

        if 'val' in metrics[0] and 'val' in metrics[1]:
            # set-wise
            plt.plot(epochs, hs[metrics[p]], 'bo', label=metrics[p])
            plt.plot(epochs, hs[metrics[p + m]], 'rx', label=metrics[p + m])
        else:
            # metric-wise
            plt.plot(epochs, hs[metrics[2 * p]], 'bo', label=metrics[2 * p])
            plt.plot(epochs, hs[metrics[2 * p + 1]], 'rx', label=metrics[2 * p + 1])

        ax.set_ylabel(ylabel_name)
        plt.grid('on')
        plt.xlim([0.5, len(epochs) + 0.5])
        plt.ylim([0, 1])
        legend = ax.legend(loc='best', prop={'size': legend_font_size})  # doesn't work
        frame = legend.get_frame()
        frame.set_facecolor('0.70')

        # Plot learning rate over epochs
        # TODO check if lr exists
        _ = add_another_y_axis(ax=ax, x=epochs, y=lr, color='g', marker='^', yscale='log', y_axis_name='Learning rate')

    ax.set_xlabel('Epochs')

    if savefig:
        if img_name:
            plt.savefig(img_name)
        else:
            plt.savefig('learning')


def add_another_y_axis(ax, x, y, color='g', marker='^', yscale='linear', y_axis_name=None):
    """ Adapted from:  https://matplotlib.org/devdocs/gallery/api/two_scales.html
    Args:
        ax (axis):  Axis to put two scales on
        x (array-like):  x-axis values for both datasets
        y (array-like):  Data for right hand scale
        color (str):  Color for added line
        marker (str):  Color for added line
        y_axis_name (str):  Name of the plotted value

    Returns:
        ax2 (axis):  New twin axis
    """
    legend_font_size = 10
    ax2 = ax.twinx()
    ax2.plot(x, y, color=color, marker=marker, label=y_axis_name)

    if y_axis_name:
        ax2.set_ylabel(y_axis_name, color=color)

    if yscale:
        ax2.set_yscale(yscale)

    for tl in ax2.get_yticklabels():
        tl.set_color(color)

    legend = ax2.legend(loc='best', prop={'size': legend_font_size})

    ymin, ymax = np.min(y)/10.0, np.max(y)*10.0
    ax2.set_ylim([ymin, ymax])
    return ax2


def filter_on_smiles_len(data, smiles_len, thres_min=None, thres_max=None):
    """ Remove SMILES based on SMILES string length if they are too short/large. """
    df = data.copy()
    if thres_max:
        df = df.loc[smiles_len < thres_max, :]
    if thres_min:
        df = df.loc[smiles_len > thres_min, :]
    return df


def general_plots(data, smiles_len):
    """ Temp function removed from main script. """
    import matplotlib.pyplot as plt

    # Plot SMILES length hist
    plt.figure()
    plt.hist(smiles_len, bins=100)
    plt.xlabel('SMILES string length')
    plt.ylabel('Count')
    plt.title('SMILES length histogram (raw) [{}, {}]'.format(smiles_len.min(), smiles_len.max()))
    plt.grid('on')
    print('\nShortest SMILES string:')
    print(data.loc[smiles_len == smiles_len.min(), :])
    print('\nLongest SMILES string:')
    print(data.loc[smiles_len == smiles_len.max(), :])
    plt.savefig('hist_smiles_len_raw')

    # Plot LCONC hist
    plt.figure()
    plt.hist(data['LCONC'], bins=30)
    plt.xlabel('LCONC')
    plt.ylabel('Count')
    plt.title('LCONC histogram (raw) [{:.2f}, {:.2f}]'.format(data['LCONC'].min(), data['LCONC'].max()))
    plt.grid('on')
    plt.savefig('hist_lconc_raw')

    # Plot GRWOTH hist
    plt.figure()
    plt.hist(data['GROWTH'], bins=30)
    plt.xlabel('GROWTH')
    plt.ylabel('Count')
    plt.title('GROWTH histogram (raw) [{:.2f}, {:.2f}]'.format(data['GROWTH'].min(), data['GROWTH'].max()))
    plt.grid('on')
    plt.savefig('hist_growth_raw')

    plt.close('all')


def discretize(y, bins=5, cutoffs=None, min_count=0, verbose=False):
    """ Takes vector y and returns a discretized version of y (y_even) with the objective to balance vector y based on
    the number of bins.
    Adapted from Fangfang's code in skwrapper.py.
    """
    thresholds = cutoffs
    if thresholds is None:
        percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
        thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    good_bins = None
    if verbose:
        bc = np.bincount(classes)
        good_bins = len(bc)
        min_y = np.min(y)
        max_y = np.max(y)
        print('Category cutoffs:', ['{:.3g}'.format(t) for t in thresholds])
        print('Bin counts:')
        for i, count in enumerate(bc):
            lower = min_y if i == 0 else thresholds[i-1]
            upper = max_y if i == len(bc)-1 else thresholds[i]
            removed = ''
            if count < min_count:
                removed = ' .. removed (<{})'.format(min_count)
                good_bins -= 1
            print('  Class {}: {:7d} ({:.4f}) - between {:+.2f} and {:+.2f}{}'.
                  format(i, count, count/len(y), lower, upper, removed))
        # print('  Total: {:9d}'.format(len(y)))
    return classes, thresholds, good_bins


def find_substr(s, substr):
    indexes = []
    idx = 0
    while idx < len(s):
        idx = s.find(substr, idx)
        if idx == -1:
            break
        # print(substr, 'found at', idx)
        indexes.append(idx)
        idx += len(substr)
    return indexes


def tokenize_smiles(samples, tokenization_method='featurize_seq_smiles'):
    """
    Description of SMILES elements (wiki, http://opensmiles.org/opensmiles.html):
    Atoms, Bonds, Rings, Aromaticity, Branching, Stereochemistry, Isotopes

    aliphatic = ['B', 'C', 'F', 'N', 'O', 'P', 'S', 'Br', 'Cl', 'I']
    aromaticity = ['b', 'c', 'n', 'o', 'p', 's'] + ['as', 'se']  # aromatic_organic and aromatic_symbols
    bond_type = ['-', '=', '#', '$', ':', '/', '\\']
    hydrogens = ['[H]', '[CH]', '[CH4]', '[ClH]', '[ClH1]']
    atom_charge = ['[N+]']
    branching = ['(', ')']
    ring_number = []  # (except charged atoms) all numbers including the ones combined with '%'
    ring_membership = []

    Dorr's paper rules:
    1. Atom type (neuron for each chemical elemenet in the dataset)
    2. Bond type (neuron for each bond type)
    3. Aromaticity (neuron for the aromaticity of atoms)
    4. Attached hydrogens (neuron for the number of attached hydrogens)
    5. Atom charge (neuron for the pos./neg. charge of an atom)
    6. Branching (neurons for the start or closure of a branch in a molecule)
    7. Ring number (2 neurons for the start and end of a cyclic structure):
       two-digit rings are preceded by '%', e.g.: C%25CCCCC%25
    8. Ring membership (neuron for the membership of an atom in a cyclic structure)

    TODO
    - the bonds indicated by \ and / always come in pairs (http://opensmiles.org/opensmiles.html), but in our dataset
      some strings contains odd number of these characters --> these are invalid as indicated in the link (??)
    - understand what is '\\' in our strings?
    """
    samples_org = samples.copy()

    # for i, s in enumerate(samples):
    #     if 'Br' in s:
    #         print('{}:  {}'.format(i, s))

    if tokenization_method == 'featurize_seq_generic':
        tokenizer = Tokenizer(num_words=None, filters='', lower=False, split='', char_level=True)
        tokenizer.fit_on_texts(texts=samples)
        sequences = tokenizer.texts_to_sequences(texts=samples)

        # Transform list of sequences into 2D numpy array
        x_data = pad_sequences(sequences=sequences, maxlen=None, padding='post', value=0)
        return x_data, tokenizer

    # =================================
    # Start manual smiles featurization
    # =================================
    # Get all elements of a priodic table
    with open('periodic_table.txt') as f:
        lines = f.read().split('\n')
    elements = [el.split()[0] for el in lines]

    # Get all elements that len(el) > 1
    elements_long = [el for el in elements if len(el) > 1]

    # Get all elements that len(el) > 1 and appear in the dataset
    elem_long = []
    for el in elements_long:
        for s in samples:
            if el in s:
                elem_long.append(el)
                break

    # Get all substrings enclosed in [...] and their count --> treat as unique elements
    # Combos of elements: [ClH-]->12 (), [NH3+]->2, [SeH]->14, [nH]->586
    br = dict()
    for i, s in enumerate(samples):
        s = '_['.join(s.split('['))
        s = ']_'.join(s.split(']'))
        s = s.split('_')
        for j in s:
            if len(j) > 0 and j[0] == '[' and j[-1] == ']':
                if j in br.keys():
                    br[j] += 1
                else:
                    br[j] = 1

    for s_id, s in enumerate(samples):
        # s = 'CC(=O)OCC(=O)[C@@]1(O)CC[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]12C'
        # s = 'COc1c(O)c2N(CCc2c3cc([nH]c13)C(=O)N4CCc5c4c(O)c(OC)c6[nH]c(cc56)C(=O)N7CC8CC89C7=CC(=O)c%10[nH]cc(C)c9%10)C(=O)N'
        # s = 'O=C1c2ccc(c3cccc(c4nc5ccccc5n14)c23)S(=O)(=O)c6ccc7C(=O)n8c(nc9ccccc89)c%10cccc6c7%10a'
        # s = 'CC1(CCC23COC4(CCC5C6(C)CCC(OC7OCC(OC8OC(CO)C(O)C(OC9OC(CO)C(O)C(O)C9O)C8OC%10OCC(O)C(O)C%10O)C(O)C7OC%11OC(CO)C(O)C(O)C%11O)C(C)(C)C6CCC5(C)C4(C)CC2O)C3C1)C=O'
        # s = 'CC(C)C[C@@H]1N2C(=O)[C@](NC(=O)[C@H]3CN(C)[C@@H]4Cc5c(Br)[nH]c6cccc(C4=C3)c56)(O[C@@]2(O)[C@@H]7CCCN7C1=O)C(C)C'; s = 'AsCKNSi' + s + 'BrBrCKNAs3Ni[Br]'
        s = '_(_'.join(s.split('('))
        s = '_)_'.join(s.split(')'))
        s = '_=_'.join(s.split('='))
        s = '_#_'.join(s.split('#'))
        s = '_['.join(s.split('['))
        s = ']_'.join(s.split(']'))
        s = '_%'.join(s.split('%'))

        # Add '_' after %##
        indexes = find_substr(s, '%')
        for i in indexes[::-1]:
            c = 1
            while i + c < len(s):
                if not s[i + c].isdigit():
                    break
                c += 1
            s = '_'.join([s[:i + c], s[i + c:]])

        # Split on '_'
        s = [i for i in s.split('_') if len(i) > 0]

        # Find elements that len(el) > 1 and split
        ss = []
        for substr in s:
            # If substring is len(substr) > 1 and it's not enclosed in [], check if it contain any of the long elements,
            # and if yes break
            if len(substr) > 1 and ('[' not in substr):
                for el in elem_long:
                    if el in substr:
                        indexes = find_substr(substr, el)
                        for i in indexes[::-1]:
                            substr = '_'.join(([substr[:i], substr[i:i + len(el)], substr[i + len(el):]]))

            ss = ss + [i for i in substr.split('_') if len(i) > 0]

        s = ss

        # Split the unsplitted sunbstrings
        k = []
        for substr in s:
            if len(substr) > 0 and (substr not in elem_long) and ('%' not in substr) and ('[' not in substr) and (']' not in substr):
                k.extend(list(substr))
            else:
                k = k + [substr]

        samples[s_id] = '_'.join(k)

    # Finally, tokenize
    tokenizer = Tokenizer(num_words=None, filters='', lower=False, split='_', char_level=False)
    tokenizer.fit_on_texts(texts=samples)

    if tokenization_method == 'featurize_seq_smiles':
        # Manual featurization (generates sequences 2-D data)
        sequences = tokenizer.texts_to_sequences(texts=samples)
        x_data = pad_sequences(sequences=sequences, maxlen=None, padding='post', value=0)

    elif tokenization_method == 'featurize_3d_smiles':
        # Manual featurization (generates 3-D data)
        token_index = tokenizer.word_index
        max_length = np.max([len(i) for i in samples_org])

        x_data = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
        for sample_id, sample in enumerate(samples):
            for word_id, word in enumerate(sample.split(sep='_')[:max_length]):
                index = token_index.get(word)
                x_data[sample_id, word_id, index] = 1.

    return x_data, tokenizer


def create_dense_model(n_features, optimizer='adam', loss='mae', initializer='glorot_uniform', metrics=None):
    """ Create densely connected model. """
    inputs = Input(shape=(n_features,), dtype=np.float32, name='inputs')
    x = layers.Dense(units=256, kernel_initializer=initializer, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=256, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=256, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=128, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=128, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=64, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=32, kernel_initializer=initializer, activation='relu')(x)
    outputs = layers.Dense(units=1, kernel_initializer=initializer)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def create_rnn_with_lconc(n_features, method='Conv1D', optimizer='adam', loss='mae', initializer='glorot_uniform',
                          metrics=None, data_dim=None, embd_input_dim=None, embd_output_dim=None):
    """ Create an rnn/conv1d net.
    This architecture takes smiles strings and lconc.

    The loss drop in rnn models is very slow! Test with alternative optimizations algs!
    """
    if data_dim == 2:
        smiles_input = Input(shape=(n_features,), dtype=np.float32, name='smiles_input')
        x = layers.Embedding(input_dim=embd_input_dim, output_dim=embd_output_dim)(smiles_input)
    elif data_dim == 3:
        smiles_input = Input(shape=(None, n_features), dtype=np.float32, name='smiles_input')
        x = smiles_input

    method = method.lower()
    if method == 'lstm':
        x = layers.LSTM(units=128, recurrent_dropout=0.1, activation='relu', return_sequences=True)(x)
        #x = layers.LSTM(units=64, recurrent_dropout=0.1, activation='relu', return_sequences=True)(x)
        smiles_output = layers.LSTM(units=128, recurrent_dropout=0.1, activation='relu')(x)
    elif method == 'gru':
        x = layers.Bidirectional(layers.GRU(32, activation='relu', return_sequences=True))(x)
        x = layers.GRU(units=64, activation='relu', return_sequences=True)(x)
        smiles_output = layers.GRU(units=64, activation='relu')(x)
    elif method == 'conv1d':
        x = layers.Conv1D(filters=256, kernel_size=13, activation='relu', kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(9)(x)
        x = layers.Conv1D(filters=256, kernel_size=13, activation='relu', kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        smiles_output = layers.GlobalMaxPooling1D()(x)

    smiles_output = layers.Dense(units=64, activation='relu')(smiles_output)
    # smiles_output = layers.BatchNormalization()(smiles_output)
    smiles_output = layers.Dense(units=32, activation='relu')(smiles_output)
    # smiles_output = layers.BatchNormalization()(smiles_output)
    smiles_output = layers.Dense(units=16, activation='relu')(smiles_output)
    # smiles_output = layers.BatchNormalization()(smiles_output)
    smiles_output = layers.Dense(units=8, activation='relu')(smiles_output)
    # smiles_output = layers.BatchNormalization()(smiles_output)

    # TODO: put a higher weight on lconc --> maybe based on output from skwrapper.py
    lconc_input = Input(shape=(1,), dtype=np.float32, name='lconc_input')

    # Concatenate smiles and lconc
    concatenated = layers.concatenate([smiles_output, lconc_input], axis=-1)

    # Add layer on top after concatenation
    x = layers.Dense(units=16, activation='relu', kernel_initializer=initializer)(concatenated)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=8, activation='relu', kernel_initializer=initializer)(x)
    output = layers.Dense(units=1, kernel_initializer=initializer)(x)

    # Instantiate model and compile
    model = Model(inputs=[smiles_input, lconc_input], outputs=[output])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model



'''
# ======================================================================================================================
# Leftovers
# ======================================================================================================================
'''
# ======================================================================================================================
# k-fold with manual split - approach 1
# ======================================================================================================================
# k_folds = 5
# kf = sklearn.model_selection.KFold(n_splits=k_folds, random_state=0)
#
# # Generate train and val indexes for k-fold CV
# tr_idx = []
# vl_idx = []
# num_val_samples = len(X) // k_folds
# for f, (train_idx, val_idx) in enumerate(kf.split(X)):
#     tr_idx.append(train_idx)
#     vl_idx.append(val_idx)
#
# # Permute the order of folds
# folds_order = np.random.permutation(range(k_folds))
# print("\nFolds order: {}".format(folds_order))
#
# # Run k-fold CV
# hs = dict()
# for f, i in enumerate(folds_order):
#     print("\nFold {} out of {}".format(f + 1, k_folds))
#     print("Train idx:  {}".format(tr_idx[i]))
#     print("Val idx:    {}".format(vl_idx[i]))
#
#     # Split the data
#     x_train, x_val = X[tr_idx[i], :], X[vl_idx[i], :]
#     y_train, y_val = Y[tr_idx[i]], Y[vl_idx[i]]
#
#     model = build_model()
#     scores = model.fit()


# ======================================================================================================================
# k-fold with manual split - approach 1
# ======================================================================================================================
# k_folds = 5
# kf = sklearn.model_selection.KFold(n_splits=k_folds, random_state=0)
#
# num_val_samples = len(X) // k_folds
# for f in range(k_folds):
#     print("\nFold {} out of {}".format(f + 1, k_folds))
#
#     # get val indexes
#     print('Val indices: [{}, {})'.format(num_val_samples * f, num_val_samples * (f + 1)))
#     tmp = np.arange(num_val_samples * f, num_val_samples * (f + 1))
#     val_idx = np.zeros((X.shape[0], ), dtype=np.bool)
#     val_idx[tmp] = True
#
#     # split train and val
#     x_train, x_val = X[~val_idx, :], X[val_idx, :]  # get train and val indexes for smiles
#     y_train, y_val = Y[~val_idx], Y[val_idx]  # get train and val indexes for lconc
#
#     model = build_model()
#     scores = model.fit()
