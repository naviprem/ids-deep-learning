from time import time
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import visuals as vs
import matplotlib.pyplot as plt
import numpy as np

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time()  # Get start time
    clf = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()  # Get end time

    # Calculate the training time
    results['train_time'] = end - start

    # Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time()  # Get start time
    predictions_test = clf.predict(X_test)
    predictions_train = clf.predict(X_train[:300])
    end = time()  # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.001, average="micro")
    # results['f_train'] = 1.0
    # # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.001, average="micro")
    # results['f_test'] = 1.0

    # Success
    print ("{} trained on {} samples. Result is {}".format(learner.__class__.__name__, sample_size, results))
    # print (confusion_matrix(y_test, predictions_test))

    cnf_matrix = confusion_matrix(y_test, predictions_test)
    np.set_printoptions(precision=2)

    # plt.figure()
    # vs.plot_confusion_matrix(cnf_matrix, classes=np.unique(y_train),
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plt.figure()
    # vs.plot_confusion_matrix(cnf_matrix, classes=np.unique(y_train), normalize=True,
    #                       title=learner.__class__.__name__ + ' - Normalized confusion matrix')
    #
    # plt.show()

    # Return the results
    return results