import load_dataset
import data_preprocess
import train_predict
from datetime import datetime
import csv

classifiers = ["GaussianNB", "LogisticRegression", "SGDClassifier", "GradientBoostingClassifier",
               "RandomForestClassifier", "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier",
               "DecisionTreeClassifier", "KNeighborsClassifier"]
for label_type in ["binary", "labels"]:
    for log_trans in [True, False]:
        for one_hot in [True, False]:
            for classifier in classifiers:
                ds = {
                    'name' : 'UNSW-NB15-tr',
                    'rs' : 9,
                    'log' : {
                        '00_log' : True,
                        '01_start_time': str(datetime.now()),
                        '02_end_time': str(datetime.now()),
                        '05_load_data_flag' : True,
                        '10_shuffle_split' : False,
                        '15_log_transform_flag' : log_trans,
                        '20_min_max_scaler_flag' : True,
                        '25_one_hot_encoding_flag' : one_hot,
                        '30_factorize_flag' : one_hot == False,
                        '35_features_count' : 0,
                        '40_pca_flag' : False,
                        '45_pca_n_components' : 0,
                        '50_label_type' : label_type,
                        '65_classifier_name' : classifier,
                        '70_hyper_parameters' : '',
                        '75_accuracy' : 0.0,
                        '76_far' : 0.0,
                        '77_fpr' : 0.0,
                        '78_fnr' : 0.0,
                        '79_train_time' : 0,
                        '80_predict_time' : 0
                    }
                }


                if ds['log']['05_load_data_flag']:
                    load_dataset.do(ds)
                    print("05_load_data_flag done")


                if ds["log"]["15_log_transform_flag"]:
                    data_preprocess.log_transform(ds)
                    print("15_log_transform_flag done")


                if ds["log"]["20_min_max_scaler_flag"]:
                    data_preprocess.min_max_scaler(ds)
                    print("20_min_max_scaler_flag done")

                if ds["log"]["25_one_hot_encoding_flag"]:
                    data_preprocess.one_hot_encode(ds)
                    print("25_one_hot_encoding_flag done")


                if ds["log"]["30_factorize_flag"]:
                    data_preprocess.factorize(ds)
                    print("30_factorize_flag done")

                ds["log"]["35_features_count"] = len(ds["X_train"].columns)

                if ds["log"]["40_pca_flag"]:
                    data_preprocess.pca(ds)
                    print("40_pca_flag done")


                train_predict.do(ds)

                print (sorted(ds['log'].keys()))
                for key in sorted(ds['log'].keys()):
                    print ("{} - {}".format(key, ds['log'][key]))
                if ds["log"]["00_log"]:
                    with open('logs.csv', 'a') as log_file:
                        writer = csv.DictWriter(log_file, sorted(ds['log'].keys()))
                        ds["log"]['02_end_time'] = str(datetime.now())
                        writer.writerow(ds['log'])