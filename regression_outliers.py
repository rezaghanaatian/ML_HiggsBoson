from implementations import least_squares_GD, least_squares_SGD, least_squares, ridge_regression
from helpers import load_csv_data, predict_labels, create_csv_submission, pre_process_data, normalize_data, \
    pre_process_data_jets, load_csv_data_general
import numpy as np
from validation import split_data, validate

data_path = "data/train.csv"
data = load_csv_data_general(data_path)
# data_features = normalize_data(data_features)

# Define the parameters of the algorithm.
# max_iterations = 1
# gamma = 0.7
# initial_w = np.ones(len(data_features[0]))
lambda_ = 0.000001

# Define Cross validation folds
folds_n = 10
validation_scores = []

for i in range(0, folds_n):

    if folds_n > 1:
        x_tr, x_te = split_data(data, 1 / folds_n)
    else:
        test_data_path = "data/test.csv"
        test_data = load_csv_data_general(test_data_path)
        x_tr = data
        x_te = test_data

    x_tr0, x_tr1, x_tr2, x_tr3, x_tr0_wm, x_tr1_wm, x_tr2_wm, x_tr3_wm = pre_process_data_jets(x_tr)
    # Start ML algorithm.
    loss0, w0 = ridge_regression(x_tr0[:, 1], x_tr0[:, 2:], lambda_)
    loss1, w1 = ridge_regression(x_tr1[:, 1], x_tr1[:, 2:], lambda_)
    loss2, w2 = ridge_regression(x_tr2[:, 1], x_tr2[:, 2:], lambda_)
    loss3, w3 = ridge_regression(x_tr3[:, 1], x_tr3[:, 2:], lambda_)
    loss0_wm, w0_wm = ridge_regression(x_tr0_wm[:, 1], x_tr0_wm[:, 2:], lambda_)
    loss1_wm, w1_wm = ridge_regression(x_tr1_wm[:, 1], x_tr1_wm[:, 2:], lambda_)
    loss2_wm, w2_wm = ridge_regression(x_tr2_wm[:, 1], x_tr2_wm[:, 2:], lambda_)
    loss3_wm, w3_wm = ridge_regression(x_tr3_wm[:, 1], x_tr3_wm[:, 2:], lambda_)

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
        # Extract prediction to upload in Kaggle
        create_csv_submission(np.concatenate((x_te0[:, 0], x_te1[:, 0], x_te2[:, 0], x_te3[:, 0], x_te0_wm[:, 0],
                                              x_te1_wm[:, 0], x_te2_wm[:, 0], x_te3_wm[:, 0])),
                              np.concatenate((y_te_predicted0, y_te_predicted1, y_te_predicted2, y_te_predicted3,
                                              y_te_predicted0_wm, y_te_predicted1_wm, y_te_predicted2_wm,
                                              y_te_predicted3_wm)),
                              "output")

cv_score = np.mean(np.array(validation_scores))
print("================ Final validation Score ================")
print("MEAN-Accuracy score:" + str(cv_score))
