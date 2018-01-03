import pandas as pd

unsw_nb15_tr = "../datasets/UNSW-NB15/UNSW_NB15_testing-set.csv"
unsw_nb15_ts = "../datasets/UNSW-NB15/UNSW_NB15_training-set.csv"
unsw_nb15_all = ["../datasets/UNSW-NB15/UNSW_NB15_1.csv", "../datasets/UNSW-NB15/UNSW_NB15_2.csv", "../datasets/UNSW-NB15/UNSW_NB15_3.csv", "../datasets/UNSW-NB15/UNSW_NB15_4.csv"]


def do(ds):
    if ds["name"] == "UNSW-NB15-tr":
        unsw_nb15_training(ds)


def unsw_nb15_training(ds):
    train = pd.read_csv(unsw_nb15_tr)
    test = pd.read_csv(unsw_nb15_ts)

    ds["X_train"] = train.drop(['id', 'attack_cat', 'label'], axis = 1)
    ds["X_test"] = test.drop(['id', 'attack_cat', 'label'], axis = 1)
    ds["y_train"] = pd.DataFrame(data = train['attack_cat'])
    ds["y_test"] = pd.DataFrame(data=test['attack_cat'])
    ds["y_train_b"] = pd.DataFrame(data=train['label'])
    ds["y_test_b"] = pd.DataFrame(data=test['label'])
    ds["y_train_e"] = pd.get_dummies(ds["y_train"])
    ds["y_test_e"] = pd.get_dummies(ds["y_test"])
    ds["y_train_f"] = pd.DataFrame(data=pd.factorize(train['attack_cat'])[0])
    ds["y_test_f"] = pd.DataFrame(data=pd.factorize(test['attack_cat'])[0])
    ds["num_keys"] = ds["X_train"].iloc[:, :-2].select_dtypes(exclude=['object']).keys()
    ds["cat_keys"] = ds["X_train"].iloc[:, :-2].select_dtypes(include=['object']).keys()