from implementations import least_squares_GD, least_squares_SGD, least_squares, ridge_regression
from helpers import load_csv_data, predict_labels, create_csv_submission, pre_process_data, normalize_data
import numpy as np
from validation import split_data, validate

data_path = "data/train.csv"
data_labels, data_features, data_ids = load_csv_data(data_path)
# outlier_cols = [4, 5, 6, 12, 26, 27, 28]
outlier_cols = []

data_features = pre_process_data(data_features, outlier_cols)
data_features = normalize_data(data_features)

# Define the parameters of the algorithm.
max_iterations = 1
gamma = 0.7
initial_w = np.ones(len(data_features[0]))
lambda_ = 0.1

# Define Cross validation folds
folds_n = 10
validation_scores = []

for i in range(0, folds_n):
    print("================ step " + str(i + 1) + " ================")

    x_tr, x_te, y_tr, y_te = split_data(data_features, data_labels, 1 / folds_n)
    # Start ML algorithm.
    loss, w = ridge_regression(y_tr, x_tr, lambda_)
    # loss, w = least_squares_SGD(y_tr, x_tr, initial_w, max_iterations, gamma)

    # Print result
    # print("================ loss ================")
    # print(loss)
    # print("================ w ================")
    # print(w)

    # Test algorithm
    y_te_predicted = predict_labels(w, x_te)
    score = validate(y_te_predicted, y_te)

    validation_scores.append(score)
    print("Accuracy score:" + str(score))

cv_score = np.mean(np.array(validation_scores))
print("================ Final validation Score ================")
print("MEAN-Accuracy score:" + str(cv_score))

loss_, w_ = ridge_regression(data_labels, data_features, lambda_)
# Generate prediction to upload in Kaggle!
test_data_path = "data/test.csv"
test_labels, test_features, test_ids = load_csv_data(test_data_path)
test_features = pre_process_data(test_features, outlier_cols)
test_predicted_labels = predict_labels(w_, test_features)

# Extract prediction results
create_csv_submission(test_ids, test_predicted_labels, "least_squares")
print("================ labels generated successfully! Finish!! ================")
