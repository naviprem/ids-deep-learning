import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def log_transform(ds):
    ds["X_train"][ds["num_keys"]] = ds["X_train"][ds["num_keys"]].apply(lambda x: np.log(x + 1))
    ds["X_test"][ds["num_keys"]] = ds["X_test"][ds["num_keys"]].apply(lambda x: np.log(x + 1))

def min_max_scaler(ds):
    scaler = MinMaxScaler()
    # print(ds["X_train"].isnull().any())
    ds["X_train"][ds["num_keys"]] = scaler.fit_transform(ds["X_train"][ds["num_keys"]])
    ds["X_test"][ds["num_keys"]] = scaler.fit_transform(ds["X_test"][ds["num_keys"]])

def one_hot_encode(ds):

    # Address training and testing dataset missmatch

    ds["X_train"].loc[ds["X_train"].state == 'no', 'state'] = "FIN"
    ds["X_train"].loc[ds["X_train"].state == 'ECO', 'state'] = "FIN"
    ds["X_train"].loc[ds["X_train"].state == 'PAR', 'state'] = "FIN"
    ds["X_train"].loc[ds["X_train"].state == 'URN', 'state'] = "FIN"

    ds["X_test"].loc[ds["X_test"].state == 'ACC', 'state'] = "FIN"
    ds["X_test"].loc[ds["X_test"].state == 'CLO', 'state'] = "FIN"

    ds["X_train"].loc[ds["X_train"].proto == 'icmp', 'proto'] = "tcp"
    ds["X_train"].loc[ds["X_train"].proto == 'rtp', 'proto'] = "tcp"

    ds["X_train"] = pd.get_dummies(ds["X_train"])
    ds["X_test"] = pd.get_dummies(ds["X_test"])

def factorize(ds):
    for key in ds["cat_keys"]:
        ds["X_train"][key] = pd.Series(data=pd.factorize(ds["X_train"][key])[0])
        ds["X_test"][key] = pd.Series(data=pd.factorize(ds["X_test"][key])[0])
    scaler = MinMaxScaler()
    ds["X_train"][ds["cat_keys"]] = scaler.fit_transform(ds["X_train"][ds["cat_keys"]])
    ds["X_test"][ds["cat_keys"]] = scaler.fit_transform(ds["X_test"][ds["cat_keys"]])

def pca(ds):
    pca = PCA(n_components=ds["pca_n_components"], random_state=ds["rs"])
    pca.fit(ds["X_train"])
    reduced_X_train = pca.transform(ds["X_train"])
    ds["X_train"] = reduced_X_train
    reduced_X_test = pca.transform(ds["X_test"])
    ds["X_test"] = reduced_X_test

def remove_outliers(ds):
    select = ["response_body_len", "dur"]
    numeric_Features = ds["X_train"][select]
    all_outliers = np.array([])
    for feature in numeric_Features.keys():
        Q1 = numeric_Features[feature].quantile(.25)
        Q3 = numeric_Features[feature].quantile(.75)
        step = (float(Q3) - float(Q1)) * 10.5

        # Display the outliers
        print("Data points considered outliers for the feature '{}':\nQ1 = {}, Q3 = {}, Step = {}".format(feature, Q1, Q3, step))
        outliers = numeric_Features[~((numeric_Features[feature] >= Q1 - step) & (numeric_Features[feature] <= Q3 + step))]
        all_outliers = np.append(all_outliers, np.array(outliers.index.values.tolist()).astype(int))
        print(numeric_Features[~((numeric_Features[feature] >= Q1 - step) & (numeric_Features[feature] <= Q3 + step))][feature])
    ds["outlier_count"] = np.count_nonzero(np.unique(all_outliers))
    print(ds["outlier_count"])
