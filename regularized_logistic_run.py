import datetime
from sklearn import linear_model

from sklearn.linear_model import SGDClassifier, LinearRegression
from implementations import least_squares_GD, least_squares_SGD
from implementations import reg_logistic_regression_GD
from helpers import load_csv_data, predict_labels, create_csv_submission, pre_process_data
import numpy as np
from validation import split_data, validate


data_path = "data/train.csv"
data_labels, data_features, data_ids = load_csv_data(data_path)
ignore_cols = []
data_features = pre_process_data(data_features, ignore_cols)
# Todo: does standardizing features help?

# Define the parameters of the algorithm.
max_iters = 1000
gamma = 0.4
initial_w = np.ones(len(data_features[0]))
lambda_ = 0.01

# Define Cross validation folds
folds_n = 10
validation_scores = []

for i in range(0, folds_n):
    print("================ step " + str(i + 1) + " ================")

    x_tr, x_te, y_tr, y_te = split_data(data_features, data_labels, 1 / folds_n)
    # Start ML algorithm.
    #loss, w = reg_logistic_regression_GD(y_tr, x_tr, initial_w, lambda_, max_iters, gamma)

    # Print result
    # print("Gradient Descent: execution time={t:.3f} seconds".format(t=execution_time))
    # print("================ loss ================")
    # print(loss)
    # print("================ w ================")
    # print(w)

    # Test algorithm
    # y_te_predicted = predict_labels(w, x_te)
    # score = validate(y_te_predicted, y_te)


    # Use SKLearn
    reg = linear_model.LinearRegression()
    reg.fit(x_tr,y_tr)
    # clf = LinearRegression(loss="squared_loss", penalty="l2", max_iter=20)
    # clf.fit(x_tr, y_tr)
    # print(clf)
    y_te_predicted_sk = reg.predict(x_te)
    score = validate(y_te_predicted_sk, y_te)



    validation_scores.append(score)
    print("Accuracy score:" + str(score))

cv_score = np.mean(np.array(validation_scores))
print("================ Final validation Score ================")
print("MEAN-Accuracy score:" + str(cv_score))



# Generate prediction to upload in Kaggle!
test_data_path = "data/test.csv"
test_labels, test_features, test_ids = load_csv_data(test_data_path)
test_predicted_labels = predict_labels(w, test_features)
# test_predicted_labels_sk = clf.predict(test_features)

# Extract prediction results
create_csv_submission(test_ids, test_predicted_labels, "regularized_logistic_predictions")
print("================ labels generated successfully! Finish!! ================")
