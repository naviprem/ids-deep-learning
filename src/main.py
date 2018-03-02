import load_dataset
import data_preprocess
import train_predict
import data_explore
from datetime import datetime
import csv

first_record = True
classifiers = ["GaussianNB", "LogisticRegression", "SGDClassifier", "GradientBoostingClassifier",
               "RandomForestClassifier", "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier",
               "DecisionTreeClassifier", "KNeighborsClassifier"]

rs = 90
nc = 11
for i in range(1):
# for rs in [39, 44, 50, 60, 90]:
# for rs in [90]:
#     for nc in [10, 11, 12, 13, 14]:
                ds = {
                    'name' : 'UNSW-NB15-tr',
                    'rs' : rs ,
                    'log' : False,
                    'start_time': str(datetime.now()),
                    'end_time': str(datetime.now()),
                    'load_data_flag' : True,
                    'load_all_flag' : False,
                    'shuffle_split_flag' : True,
                    'remove_outliers_flag': False,
                    'log_transform_flag' : True,
                    'data_exploration_flag' : False,
                    'min_max_scaler_flag' : True,
                    'one_hot_encoding_flag' : False,
                    'fit_flag' : True,
                    'features_count' : 0,
                    'pca_flag' : False,
                    'pca_n_components' : nc,
                    'label_type' : "binary_encoded",
                    'cross_validation_flag' : False,
                    'classifier_name' : "ANN-01",
                    # "hyper_parameters" : {'max_features': ["sqrt", "log2", 35, 40], 'criterion': ["gini", "entropy"],
                    #               'min_impurity_decrease': [0.00000001, 0.0000001, 0.000001]},
                    # "hyper_parameters": {'max_features': ["sqrt", "log2", 35, 40]},
                    "hyper_parameters" : {'max_features':['log2'],'criterion': ["entropy"]},
                    "parameters": {'max_features': 'auto', 'criterion': "entropy", 'min_samples_split' : 2, 'random_state' : rs},
                    'accuracy' : 0,
                    'far' : 0,
                    'fpr' : 0,
                    'fnr' : 0,
                    'train_time' : 0,
                    'predict_time' : 0,
                    'log_keys' : ["start_time", "end_time", "shuffle_split_flag", "remove_outlier_flag",
                                  "outlier_count", "log_transform_flag",
                                  "min_max_scaler_flag", "one_hot_encoding_flag", "features_count",
                                  "pca_flag", "pca_n_components", "label_type", "classifier_name", "accuracy",
                                  "far", "fpr", "fnr", "train_time", "predict_time", "parameters", "hyper_parameters"]
                }


                if ds['load_data_flag']:
                    load_dataset.subset(ds)
                elif ds['load_all_flag']:
                    load_dataset.all(ds)

                if ds["log_transform_flag"]:
                    data_preprocess.log_transform(ds)

                if ds["data_exploration_flag"]:
                    data_explore.do(ds)

                if ds["remove_outliers_flag"]:
                    ds["outlier_count"] = data_preprocess.remove_outliers(ds)

                if ds["min_max_scaler_flag"]:
                    data_preprocess.min_max_scaler(ds)

                if ds["one_hot_encoding_flag"]:
                    data_preprocess.one_hot_encode(ds)
                else:
                    data_preprocess.factorize(ds)

                ds["features_count"] = len(ds["X_train"].columns)

                if ds["pca_flag"]:
                    data_preprocess.pca(ds)

                if ds["fit_flag"]:
                    train_predict.do(ds)

                    if ds["log"]:

                        with open('logs.csv', 'a') as log_file:
                            if first_record:
                                wr = csv.writer(log_file)
                                wr.writerow(ds["log_keys"])
                            writer = csv.DictWriter(log_file, ds["log_keys"], extrasaction="ignore")
                            ds['end_time'] = str(datetime.now())
                            print (ds['rs'], ds['pca_n_components'], ds["accuracy"])
                            writer.writerow(ds)
                        with open('feature-selection.csv', 'a') as feature_file:
                            wr = csv.writer(feature_file)
                            if first_record:
                                # wr.writerow(ds["X_train"].columns.values)
                                first_record = False
                            wr.writerow(ds["feature-selection"])