from time import time
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_classifier(classifier_name):
    if classifier_name == "GaussianNB":
        return GaussianNB()
    elif classifier_name == "LogisticRegression":
        return LogisticRegression()
    elif classifier_name == "SGDClassifier":
        return SGDClassifier()
    elif classifier_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier()
    elif classifier_name == "RandomForestClassifier":
        return RandomForestClassifier()
    elif classifier_name == "AdaBoostClassifier":
        return AdaBoostClassifier()
    elif classifier_name == "BaggingClassifier":
        return BaggingClassifier()
    elif classifier_name == "ExtraTreesClassifier":
        return ExtraTreesClassifier()
    elif classifier_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier()
    elif classifier_name == "KNeighborsClassifier":
        return KNeighborsClassifier()

def get_label_type(ds):
    if ds["log"]["50_label_type"] == "factorized_labels":
        return "_f"
    if ds["log"]["50_label_type"] == "labels":
        return ""
    if ds["log"]["50_label_type"] == "binary":
        return "_b"

def do(ds):
    classifier = get_classifier(ds["log"]["65_classifier_name"])

    start = time()  # Get start time
    clf = classifier.fit(ds["X_train"], ds["y_train" + get_label_type(ds)])
    end = time()  # Get end time
    ds['log']['79_train_time'] = end - start

    start = time()  # Get start time
    p_test = clf.predict(ds["X_test"])
    end = time()  # Get end time
    ds['log']['80_predict_time'] = end - start

    ds['log']['75_accuracy'] = accuracy_score(ds["y_test" + get_label_type(ds)], p_test)




    # cnf_matrix = confusion_matrix(y_test, predictions_test)
    # np.set_printoptions(precision=2)

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
