import os
import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras import Input
from keras import layers, optimizers, losses, regularizers
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, LSTM, GRU, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model


def r_square(y_true, y_pred):
    """ R^2 learning metric for keras model.fit(). """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r_sqr = (1.0 - SS_res/(SS_tot + K.epsilon()))
    return r_sqr


def r_square_adjusted(y_true, y_pred, n, k):
    """ Radj^2 learning metric for keras model.fit().
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


def get_performance_metrics(history):
    """ Extract names of all the recorded performance metrics from keras `history` variable for train and val sets.
    The performance metrics can be indentified as those that start with 'val'.
    """
    all_metrics = list(history.history.keys())  # all metrics including everything returned from callbacks
    pr_metrics = []  # performance metrics recorded for train and val such as 'loss', etc. (excluding callbacks)
    for m in all_metrics:
        if 'val' in m:
            pr_metrics.append('_'.join(m.split('_')[1:]))

    return pr_metrics


def plot_learning_kfold(hs, savefig=True, img_name='learn_kfold'):
    """ Plot the learning progress (averaged across k folds).

    Args:
        hs (dict of keras callbacks): a callback object from keras model tranining model.fit()
        savefig (bool): wether to save the fig
        img_name (figure name): name of the saved figure
    """
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [10, 8]
    legend_font_size = 10

    epochs = np.asarray(hs[0].epoch) + 1
    k_folds = len(hs)

    # Extract names of all recorded performance metrics for training and val sets
    pr_metrics = get_performance_metrics(hs[0])

    # Plot
    for m in pr_metrics:
        metric_name = m
        metric_name_val = 'val_' + m

        # Compute the average of a metric across folds
        metric_avg = np.asarray([hs[fold].history[metric_name] for fold in hs]).sum(axis=0, keepdims=True) / k_folds
        metric_avg_val = np.asarray([hs[fold].history[metric_name_val] for fold in hs]).sum(axis=0, keepdims=True) / k_folds

        # Plot a metric for each fold vs epochs
        # TODO: check how this code can be combined with plot_learning()
        marker = ['b.', 'r^', 'kx', 'mv', 'gp', 'bs', 'r8', 'kD']
        fig = plt.figure()
        for i, metric in enumerate([metric_name, metric_name_val]):
            ax = fig.add_subplot(3, 1, i + 1)
            for fold in range(k_folds):
                plt.plot(epochs, hs[fold].history[metric], label="fold{}".format(fold + 1))
            plt.ylabel(metric)
            plt.grid('on')
            plt.xlim([0.5, len(epochs) + 0.5])
            plt.ylim([0, 1])
            plt.legend(loc='best', prop={'size': legend_font_size})

        # Plot the average of a metric across folds vs epochs
        ax = fig.add_subplot(3, 1, 3)
        plt.plot(epochs, metric_avg.flatten(), 'bo', label=metric_name)
        plt.plot(epochs, metric_avg_val.flatten(), 'rx', label=metric_name_val)
        plt.ylabel(metric_name+' avg over folds')
        plt.xlabel('Epochs')
        plt.grid('on')
        plt.xlim([0.5, len(epochs) + 0.5])
        plt.ylim([0, 1])
        plt.legend(loc='best', prop={'size': legend_font_size})

        if savefig:
            plt.savefig(img_name + '_' + metric_name + '.png')


def plot_learning(history, savefig=True, img_name='learning_with_lr'):
    """ Plot the learning progress for all the recorded metrics. This function should be used with hold-out validation
    scheme since it allows to plot learning rate on a separate axis.

    Args:
        history (keras callback): return callback object from keras model tranining model.fit()
        savefig (bool): wether to save the fig
        img_name (figure name): name of the saved figure
    """
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [10, 8]
    legend_font_size = 10

    # Get the epochs vector and the recorded metrics during training
    epochs = np.asarray(history.epoch) + 1
    hh = history.history.copy()

    # Extract names of all recorded performance metrics for training and val sets
    pr_metrics = get_performance_metrics(history)

    fig = plt.figure()
    for p, m in enumerate(pr_metrics):
        ax = fig.add_subplot(len(pr_metrics), 1, p + 1)

        metric_name = m
        metric_name_val = 'val_' + m

        plt.plot(epochs, hh[metric_name], 'bo', label=metric_name)
        plt.plot(epochs, hh[metric_name_val], 'rx', label=metric_name_val)
        plt.ylabel(metric_name)

        plt.grid('on')
        plt.xlim([0.5, len(epochs) + 0.5])
        plt.ylim([0, 1])
        legend = ax.legend(loc='upper left', prop={'size': legend_font_size})
        frame = legend.get_frame()
        frame.set_facecolor('0.70')

        # Plot learning rate over epochs
        if 'lr' in hh.keys():
            _ = add_another_y_axis(ax=ax, x=epochs, y=hh['lr'], color='g', marker='^', yscale='log', y_axis_name='Learning Rate')

    ax.set_xlabel('Epochs')

    if savefig:
        plt.savefig(img_name)


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

    legend = ax2.legend(loc='upper right', prop={'size': legend_font_size})

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
    """ Find all the occurances of a substring `substr` in `s`.

    Args:
        s (str): the string
        substr (str): the substring
    Returns:
        indexes (int): Indixes where the substring `substr` start with respect to string `s`
    """
    assert len(s) >= len(substr), "The string `s`={} cannot be shorter than `substr`={} in find_substr(). ".format(s, substr)
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


def tokenize_smiles(samples, token_method='seq_smiles'):
    """
    Args:
        samples (list of strings): list of SMILES strings
        tokenization_method: determines how the SMILES strings are tokenize
            method1 ('featurize_seq_generic'):
            method2 ('featurize_seq_smiles'):
            method3 (featurize_3d_smiles'):

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
      some strings contain odd number of these characters --> these are invalid as indicated in the link (??)
    - understand what is '\\' in our strings?
    """
    samples_org = samples.copy()

    # Check if a specific symbol appears in the SMILES dataset
    # symbol = 'Br'
    # for i, s in enumerate(samples):
    #     if symbol in s:
    #         print('{}:  {}'.format(i, s))

    if token_method == 'seq_generic':
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

        # Split the unsplitted substrings
        k = []
        for substr in s:
            if len(substr) > 0 and (substr not in elem_long) and ('%' not in substr)\
                    and ('[' not in substr) and (']' not in substr):
                k.extend(list(substr))
            else:
                k = k + [substr]

        samples[s_id] = '_'.join(k)

    # Finally, tokenize
    tokenizer = Tokenizer(num_words=None, filters='', lower=False, split='_', char_level=False)
    tokenizer.fit_on_texts(texts=samples)

    if token_method == 'seq_smiles':
        # Manual featurization (generates sequences 2-D data)
        sequences = tokenizer.texts_to_sequences(texts=samples)
        x_data = pad_sequences(sequences=sequences, maxlen=None, padding='post', value=0)

    elif token_method == '3d_smiles':
        # Manual featurization (generates 3-D data)
        token_index = tokenizer.word_index
        max_length = np.max([len(i) for i in samples_org])

        x_data = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
        for sample_id, sample in enumerate(samples):
            for word_id, word in enumerate(sample.split(sep='_')[:max_length]):
                index = token_index.get(word)
                x_data[sample_id, word_id, index] = 1.0

    return x_data, tokenizer


def save_results(train_scores, val_scores, model_name):
    """ Save training results collected with k-fold cv into file.
    Args:
        train_scores & val_scores (pandas dataframe): each row value represents a train/val score for each epoch;
            the column names are the recorded performance metrics (e.g., loss, mae, r_square)
    """
    metrics = list(train_scores.columns)
    k_folds = len(train_scores)

    scores_fname = "{}.scores".format(model_name)

    with open(scores_fname, "w") as scores_file:
        for i, m in enumerate(metrics):
            scores_file.write("{:<13}Train,  Val\n".format(m))
            for f in range(len(val_scores)):
                scores_file.write("  fold {}/{}:  {:=+5.3f}, {:=+5.3f}\n".format(
                    f + 1, k_folds, train_scores.iloc[f, i], val_scores.iloc[f, i]))
            scores_file.write("\n")

        scores_file.write("{:<15}Train,  Val\n".format(''))
        for i, m in enumerate(metrics):
            scores_file.write("Mean {:<10}{:=+5.3f}, {:=+5.3f}\n".format(m,
                              train_scores.iloc[:, i].sum() / k_folds, val_scores.iloc[:, i].sum() / k_folds))


def print_results(train_scores, val_scores):
    """ Print training results collected with k-fold cv into file.
    Args:
        train_scores & val_scores (pandas dataframe): each row value represents a train/val score for each epoch;
            the column names are the recorded performance metrics (e.g., loss, mae, r_square)
    """
    metrics = list(train_scores.columns)
    k_folds = len(train_scores)

    for i, m in enumerate(metrics):
        print("\n{:<13}Train,  Val".format(m))
        for f in range(len(val_scores)):
            print("  fold {}/{}:  {:=+5.3f}, {:=+5.3f}".format(f + 1, k_folds, train_scores.iloc[f, i],
                                                                               val_scores.iloc[f, i]))

    print("\n{:<15}Train,  Val".format(''))
    for i, m in enumerate(metrics):
        print("Mean {:<10}{:=+5.3f}, {:=+5.3f}".format(m, train_scores.iloc[:, i].sum() / k_folds,
                                                          val_scores.iloc[:, i].sum() / k_folds))


def create_dense_model(input_shape, optimizer='adam', loss='mae', initializer='glorot_uniform', metrics=None):
    """ Create densely connected model. """
    inputs = Input(shape=input_shape, dtype=np.float32, name='inputs')
    x = layers.Dense(units=512, kernel_initializer=initializer, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=256, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=128, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=64, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=32, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(units=1, kernel_initializer=initializer)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def create_rnn_with_lconc(input_shape, layer='Conv1D', optimizer='adam', loss='mae', initializer='glorot_uniform',
                          metrics=None, data_dim=None, embed_input_dim=None, embed_output_dim=None,
                          model_fig_name='model'):
    """ Create a sequence learning NN (RNN or Conv1D).
    This architecture takes SMILES strings and LCONC.
    """
    # Define input layer for SMILES data
    if data_dim == 2:
        smiles_input = Input(shape=input_shape, dtype=np.float32, name='smiles_input')
        x = Embedding(input_dim=embed_input_dim, output_dim=embed_output_dim)(smiles_input)
    elif data_dim == 3:
        smiles_input = Input(shape=input_shape, dtype=np.float32, name='smiles_input')
        x = smiles_input

    # Define input layer for LCONC
    lconc_input = Input(shape=(1,), dtype=np.float32, name='lconc_input')

    # Define the sequence processing layers for SMILES data
    layer = layer.lower()
    if layer == 'lstm':
        x = LSTM(units=32, recurrent_dropout=0.1, activation='relu', return_sequences=True)(x)
        # x = LSTM(units=32, recurrent_dropout=0.1, activation='relu', return_sequences=True)(x)
        x = LSTM(units=32, recurrent_dropout=0.1, activation='relu')(x)

    elif layer == 'gru':
        # x = Bidirectional(GRU(units=32, activation='relu', return_sequences=True))(x)
        x = GRU(units=32, activation='relu', return_sequences=True)(x)
        x = GRU(units=32, activation='relu')(x)

    elif layer == 'conv1d':
        x = Conv1D(filters=64, kernel_size=21, activation='relu', kernel_initializer=initializer)(x)
        x = MaxPooling1D(pool_size=9)(x)
        x = Conv1D(filters=64, kernel_size=21, activation='relu', kernel_initializer=initializer)(x)
        x = MaxPooling1D(pool_size=9)(x)
        # x = Conv1D(filters=64, kernel_size=13, activation='relu', kernel_initializer=initializer)(x)
        # x = MaxPooling1D(pool_size=3)(x)
        # smiles_output = GlobalMaxPooling1D()(x)
        x = Flatten()(x)

    # Layers after SMILES processing
    x = Dense(units=256, activation=None, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(units=256, activation=None, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(units=256, activation=None, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    smiles_output = Activation('relu')(x)

    # Concatenate SMILES and LCONC
    concatenated = layers.concatenate([smiles_output, lconc_input], axis=-1)

    # Layers after concatenation
    x = Dense(units=256, activation=None, kernel_initializer=initializer)(concatenated)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(units=256, activation=None, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    # Output layer
    output = Dense(units=1, kernel_initializer=initializer)(x)

    # Instantiate model and compile
    model = Model(inputs=[smiles_input, lconc_input], outputs=[output])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Save model schematic (building blocks)
    plot_model(model, to_file=model_fig_name+'.png', show_shapes=True)
    return model


