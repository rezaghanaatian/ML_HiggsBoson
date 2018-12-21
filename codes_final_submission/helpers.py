# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def load_csv_data_general(data_path):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    x[np.where(y == 'b'), 1] = -1
    x[np.where(y == 's'), 1] = 1

    return x


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    data = np.zeros((len(ids), 2))
    data[:, 0] = ids
    data[:, 1] = y_pred
    data.view('i8,i8').sort(order=['f0'], axis=0)
    output_path = "outputs/"
    if not name.endswith(".csv"):
        name += ".csv"
    with open(output_path + name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in data:
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def normalize_data(data):
    """normalize the data by (x - mean(x)) / std(x)."""
    mean_data = np.mean(data)
    data = data - mean_data
    std_data = np.std(data)
    data = data / std_data
    return data


def normalize_data_features(data):
    """normalize the data by (x - mean(x)) / std(x)."""
    PRI_jet_num_index = 24
    DER_mass_MMC_index = 2
    jets = data[:, PRI_jet_num_index].copy()
    MMCs = data[:, DER_mass_MMC_index].copy()
    mean_data = np.mean(data[:, 2:])
    data[:, 2:] = data[:, 2:] - mean_data
    std_data = np.std(data[:, 2:])
    data[:, 2:] = data[:, 2:] / std_data
    data[:, PRI_jet_num_index] = jets[:]
    data[:, DER_mass_MMC_index] = MMCs[:]

    return data


def pre_process_data_jets(data):
    """
    Remove outliers based on PRI_jet_num.
    Arguments: data_features
    """
    PRI_jet_num_index = 24
    DER_mass_MMC_index = 2
    outliers_cols_jet0 = [6, 7, 14, 28, 29, 30, 21, 22, 23]
    outliers_cols_jet1 = [6, 7, 14, 28, 29, 30]
    outliers_cols_jet2 = []
    outliers_cols_jet3 = []

    outlier = -999

    jet0 = []
    jet1 = []
    jet2 = []
    jet3 = []
    jet0_wm = []
    jet1_wm = []
    jet2_wm = []
    jet3_wm = []

    jet0_indexes = []
    jet1_indexes = []
    jet2_indexes = []
    jet3_indexes = []

    for i in range(len(data[0])):
        if i not in outliers_cols_jet0:
            jet0_indexes.append(i)

        if i not in outliers_cols_jet1:
            jet1_indexes.append(i)

        if i not in outliers_cols_jet2:
            jet2_indexes.append(i)

        if i not in outliers_cols_jet3:
            jet3_indexes.append(i)

    jet0_wm_indexes = jet0_indexes.copy()
    jet1_wm_indexes = jet1_indexes.copy()
    jet2_wm_indexes = jet2_indexes.copy()
    jet3_wm_indexes = jet3_indexes.copy()

    jet0_wm_indexes.remove(DER_mass_MMC_index)
    jet1_wm_indexes.remove(DER_mass_MMC_index)
    jet2_wm_indexes.remove(DER_mass_MMC_index)
    jet3_wm_indexes.remove(DER_mass_MMC_index)

    for i in range(len(data)):
        row = data[i]
        if row[PRI_jet_num_index] == 0:
            if row[DER_mass_MMC_index] == outlier:
                revised_row = row[jet0_wm_indexes]
                jet0_wm.append(revised_row)
            else:
                revised_row = row[jet0_indexes]
                jet0.append(revised_row)
        elif row[PRI_jet_num_index] == 1:
            if row[DER_mass_MMC_index] == outlier:
                revised_row = row[jet1_wm_indexes]
                jet1_wm.append(revised_row)
            else:
                revised_row = row[jet1_indexes]
                jet1.append(revised_row)
        elif row[PRI_jet_num_index] == 2:
            if row[DER_mass_MMC_index] == outlier:
                revised_row = row[jet2_wm_indexes]
                jet2_wm.append(revised_row)
            else:
                revised_row = row[jet2_indexes]
                jet2.append(revised_row)
        else:
            if row[DER_mass_MMC_index] == outlier:
                revised_row = row[jet3_wm_indexes]
                jet3_wm.append(revised_row)
            else:
                revised_row = row[jet3_indexes]
                jet3.append(revised_row)

    return np.array(jet0), np.array(jet1), np.array(jet2), np.array(jet3), np.array(jet0_wm), np.array(
        jet1_wm), np.array(jet2_wm), np.array(jet3_wm)


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


def pre_process_data(data_features, ignore_cols):
    """
    Substitute -999 values with median of column and remove custom columns
    Arguments: data_features
               ignore_cols (list of columns should be ignored)
    """
    removed = 0
    for col in ignore_cols:
        data_features = np.delete(data_features, col - removed, 1)
        removed += 1

    columns_size = len(data_features[0])
    rows_size = len(data_features)

    for i in range(columns_size):
        x = remove_values_from_list(data_features[:, i], -999)
        median = np.median(x)
        print(str(i) + "---" + str(median))
        for j in range(rows_size):
            if data_features[j, i] == -999:
                data_features[j, i] = median
    return data_features
