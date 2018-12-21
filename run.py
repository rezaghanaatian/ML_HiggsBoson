from implementations import least_squares_GD, least_squares_SGD, least_squares, ridge_regression
from helpers import load_csv_data, predict_labels, create_csv_submission, pre_process_data, normalize_data_features, \
    pre_process_data_jets, load_csv_data_general
import numpy as np
from validation import split_data, validate

# ======================= Reading training data ======================\

data_path = "data/train.csv"
data = load_csv_data_general(data_path)

# Data normalization (standardization)
#data_normalized = normalize_data_features(data.copy())   # Decreased the accuracy!!! so we do not use it!

# ======================= Tune hyperparameters ======================\

# NOTE: This part of the code is commented out so that our Kaggle results are reproduced. (Please see the next part.)
# Hyperparameters fine-tuned using this bit of the code are quoted below.

# def find_best_lambda(data):
#     folds_n = 10
#     parameters = [1, 0.9, 0.7, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
#     param_results = []
#     for param in parameters:
#         validation_scores = []
#         for i in range(0, folds_n):
#             x_tr, x_te = split_data(data, 1 / folds_n)
#
#             # Start ML algorithm.
#             loss, w = ridge_regression(x_tr[:, 1], x_tr[:, 2:], param)
#
#             # Test algorithm
#             y_te_predicted = predict_labels(w, x_te[:, 2:])
#             score = validate(y_te_predicted, x_te[:, 1])
#             validation_scores.append(score)
#
#         cv_score = np.mean(np.array(validation_scores))
#         param_results.append(cv_score)
#
#     return parameters[param_results.index(np.max(np.array(param_results)))]
#
# # Find best hyper parameters of the algorithm.
# x_tr0, x_tr1, x_tr2, x_tr3, x_tr0_wm, x_tr1_wm, x_tr2_wm, x_tr3_wm = pre_process_data_jets(data)
#
# lambda0 = find_best_lambda(x_tr0)
# lambda0_wm = find_best_lambda(x_tr0_wm)
# lambda1 = find_best_lambda(x_tr1)
# lambda1_wm = find_best_lambda(x_tr1_wm)
# lambda2 = find_best_lambda(x_tr2)
# lambda2_wm = find_best_lambda(x_tr2_wm)
# lambda3 = find_best_lambda(x_tr3)
# lambda3_wm = find_best_lambda(x_tr3_wm)
#
# print("lambda0: " + str(lambda0))
# print("lambda0_wm: " + str(lambda0_wm))
# print("lambda1: " + str(lambda1))
# print("lambda1_wm: " + str(lambda1_wm))
# print("lambda2: " + str(lambda2))
# print("lambda2_wm: " + str(lambda2_wm))
# print("lambda3: " + str(lambda3))
# print("lambda3_wm: " + str(lambda3_wm))



# ==== Best lambdas found as in our Kaggle submission ====\
# These values are hardcoded only with the purpose of reproducibility of exact Kaggle results.
lambda0 = 0.0001
lambda0_wm = 1
lambda1 = 1e-06
lambda1_wm = 0.0001
lambda2 = 0.001
lambda2_wm = 0.5
lambda3 = 0.01
lambda3_wm = 0.5


# Cross validation
folds_n = 1    # Set this to 1 to generate our output for Kaggle! If you want to do local cv set this to a number (e.g. 10)
validation_scores = []


# ======================= Train the algorithm on training set ======================\

for i in range(0, folds_n):

    if folds_n > 1:
        x_tr, x_te = split_data(data, 1 / folds_n)
    else:
        test_data_path = "data/test.csv"
        test_data = load_csv_data_general(test_data_path)
        x_tr = data
        x_te = test_data

    # Split the training set according to the rules cited in our report
    x_tr0, x_tr1, x_tr2, x_tr3, x_tr0_wm, x_tr1_wm, x_tr2_wm, x_tr3_wm = pre_process_data_jets(x_tr)

    # Start ML algorithm - train using ridge regression algorithm
    loss0, w0 = ridge_regression(x_tr0[:, 1], x_tr0[:, 2:], lambda0)
    loss1, w1 = ridge_regression(x_tr1[:, 1], x_tr1[:, 2:], lambda1)
    loss2, w2 = ridge_regression(x_tr2[:, 1], x_tr2[:, 2:], lambda2)
    loss3, w3 = ridge_regression(x_tr3[:, 1], x_tr3[:, 2:], lambda3)
    loss0_wm, w0_wm = ridge_regression(x_tr0_wm[:, 1], x_tr0_wm[:, 2:], lambda0_wm)
    loss1_wm, w1_wm = ridge_regression(x_tr1_wm[:, 1], x_tr1_wm[:, 2:], lambda1_wm)
    loss2_wm, w2_wm = ridge_regression(x_tr2_wm[:, 1], x_tr2_wm[:, 2:], lambda2_wm)
    loss3_wm, w3_wm = ridge_regression(x_tr3_wm[:, 1], x_tr3_wm[:, 2:], lambda3_wm)

    # Test algorithm
    x_te0, x_te1, x_te2, x_te3, x_te0_wm, x_te1_wm, x_te2_wm, x_te3_wm = pre_process_data_jets(x_te)
    y_te_predicted0 = predict_labels(w0, x_te0[:, 2:])
    y_te_predicted1 = predict_labels(w1, x_te1[:, 2:])
    y_te_predicted2 = predict_labels(w2, x_te2[:, 2:])
    y_te_predicted3 = predict_labels(w3, x_te3[:, 2:])
    y_te_predicted0_wm = predict_labels(w0_wm, x_te0_wm[:, 2:])
    y_te_predicted1_wm = predict_labels(w1_wm, x_te1_wm[:, 2:])
    y_te_predicted2_wm = predict_labels(w2_wm, x_te2_wm[:, 2:])
    y_te_predicted3_wm = predict_labels(w3_wm, x_te3_wm[:, 2:])

    if folds_n > 1:
            score0 = validate(y_te_predicted0, x_te0[:, 1])
            score1 = validate(y_te_predicted1, x_te1[:, 1])
            score2 = validate(y_te_predicted2, x_te2[:, 1])
            score3 = validate(y_te_predicted3, x_te3[:, 1])
            score0_wm = validate(y_te_predicted0_wm, x_te0_wm[:, 1])
            score1_wm = validate(y_te_predicted1_wm, x_te1_wm[:, 1])
            score2_wm = validate(y_te_predicted2_wm, x_te2_wm[:, 1])
            score3_wm = validate(y_te_predicted3_wm, x_te3_wm[:, 1])

            # validation_scores.append(score)
            print("Accuracy score-jet0:" + str(score0))
            print("Accuracy score-jet1:" + str(score1))
            print("Accuracy score-jet2:" + str(score2))
            print("Accuracy score-jet3:" + str(score3))
            print("Accuracy score-jet0_wm:" + str(score0_wm))
            print("Accuracy score-jet1_wm:" + str(score1_wm))
            print("Accuracy score-jet2_wm:" + str(score2_wm))
            print("Accuracy score-jet3_wm:" + str(score3_wm))
            final_score = (score0 * len(y_te_predicted0) + score1 * len(y_te_predicted1) + score2 * len(
                y_te_predicted2) + score3 * len(y_te_predicted3) +
                           score0_wm * len(y_te_predicted0_wm) + score1_wm * len(y_te_predicted1_wm) + score2_wm * len(
                y_te_predicted2_wm) + score3_wm * len(y_te_predicted3_wm)) / (
                              len(y_te_predicted0) + len(y_te_predicted1) + len(y_te_predicted2) + len(y_te_predicted3) +
                              len(y_te_predicted0_wm) + len(y_te_predicted1_wm) + len(y_te_predicted2_wm) + len(
                                  y_te_predicted3_wm))

            print("================ step " + str(i + 1) + " : " + str(final_score) + " ================")

            validation_scores.append(final_score)
    else:
        # Extract predictions file to upload on Kaggle
        create_csv_submission(np.concatenate((x_te0[:, 0], x_te1[:, 0], x_te2[:, 0], x_te3[:, 0], x_te0_wm[:, 0],
                                              x_te1_wm[:, 0], x_te2_wm[:, 0], x_te3_wm[:, 0])),
                              np.concatenate((y_te_predicted0, y_te_predicted1, y_te_predicted2, y_te_predicted3,
                                              y_te_predicted0_wm, y_te_predicted1_wm, y_te_predicted2_wm,
                                              y_te_predicted3_wm)),
                              "output")

if folds_n > 1:
    cv_score = np.mean(np.array(validation_scores))
    print("================ Final validation Score ================")
    print("MEAN-Accuracy score:" + str(cv_score))

else:
    print("================ Output generated successfully in output folder! ================")