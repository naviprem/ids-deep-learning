from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

np.random.seed(42)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger


def get_classifier(classifier_name, ds):
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
    elif classifier_name == "ANN-01":
        return construct_ann_01()

def construct_ann_01():


    model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(42,)))
    # model.add(Dropout(.6))
    # model.add(Dense(384, activation='relu', input_shape=(42,)))
    # model.add(Dropout(.62))
    model.add(Dense(256, activation='relu', input_shape=(42,)))
    model.add(Dropout(.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(2, activation='softmax'))

    # Compiling the model
    model.compile(loss = 'categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    model.summary()
    return model
    # model.fit(X_train_3d, ye_train, epochs=2, batch_size=100, verbose=0)
    #
    # score = model.evaluate(X_train_3d, ye_train)
    # print("\n Training Accuracy:", score)
    # score = model.evaluate(X_test_3d, ye_test)
    # print("\n Testing Accuracy:", score)

def get_label_type(ds):
    if ds["label_type"] == "factorized_labels":
        return "_f"
    elif ds["label_type"] == "labels":
        return ""
    elif ds["label_type"] == "binary":
        return "_b"
    elif ds["label_type"] == "binary_encoded":
        return "_be"
    elif ds["label_type"] == "encoded":
        return "_e"

def do(ds):
    if ds['shuffle_split_flag']:
        ds["X_train"], ds["y_train" + get_label_type(ds)] = shuffle(ds["X_train"], ds["y_train" + get_label_type(ds)])
        # X_train, X_valid, y_train, y_valid = train_test_split(ds["X_train"],
        #                                                     ds["y_train" + get_label_type(ds)],
        #                                                     test_size=0.2,
        #                                                     random_state=9)

    classifier = get_classifier(ds["classifier_name"], ds)
    if ds['cross_validation_flag']:
        start = time()  # Get start time
        parameters = ds["hyper_parameters"]
        scorer = make_scorer(accuracy_score)
        grid_obj = GridSearchCV(classifier, param_grid=parameters, scoring=scorer)
        grid_fit = grid_obj.fit(ds["X_train"], ds["y_train" + get_label_type(ds)])
        end = time()  # Get end time
        ds['train_time'] = end - start

        best_clf = grid_fit.best_estimator_
        ds["best_clf"] = best_clf
        ds["feature-selection"] = best_clf.feature_importances_

        start = time()  # Get start time
        best_predictions = best_clf.predict(ds["X_test"])
        if ds["label_type"] == "labels":
            best_predictions = [0 if l == "Normal" else 1 for l in best_predictions]
        ds['accuracy'] = float(accuracy_score(ds["y_test_b"], pd.DataFrame(data=best_predictions)))
        end = time()  # Get end time
        ds['predict_time'] = end - start
    else:
        if ds['classifier_name'] not in ['ANN-01']:
            classifier.set_params(**ds["parameters"])

            start = time()  # Get start time
            clf = classifier.fit(ds["X_train"], ds["y_train" + get_label_type(ds)])
            end = time()  # Get end time
            ds['train_time'] = end - start

            ds["feature-selection"] = clf.feature_importances_

            start = time()  # Get start time
            p_test = clf.predict(ds["X_test"])
            end = time()  # Get end time
            ds['predict_time'] = end - start

            if ds["label_type"] == "labels":
                p_test = [0 if l == "Normal" else 1 for l in p_test]
            ds['accuracy'] = float(accuracy_score(ds["y_test_b"], pd.DataFrame(data=p_test)))
        else:

            checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
                                           save_best_only=True, monitor='val_acc', mode='max')
            earlyetopping = EarlyStopping(monitor='val_acc', patience=50, verbose=1, mode='max')
            csvlogger = CSVLogger("keras-logger.csv", separator=',', append=True)
            classifier.fit(ds["X_train"], ds["y_train" + get_label_type(ds)], epochs=500, batch_size=3500,
                           validation_split=0.1, verbose=2,
                           callbacks=[checkpointer, earlyetopping, csvlogger], shuffle=True)
            classifier.load_weights('model.weights.best.hdf5')
            score = classifier.evaluate(ds["X_test"], ds["y_test" + get_label_type(ds)])
            print("\n Testing Accuracy:", score)



