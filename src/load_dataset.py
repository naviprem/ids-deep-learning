import pandas as pd

unsw_nb15_tr = "../datasets/UNSW-NB15/UNSW_NB15_testing-set.csv"
unsw_nb15_ts = "../datasets/UNSW-NB15/UNSW_NB15_training-set.csv"
unsw_nb15_1 = "../datasets/UNSW-NB15/UNSW-NB15_1.csv"
unsw_nb15_2 = "../datasets/UNSW-NB15/UNSW-NB15_2.csv"
unsw_nb15_3 = "../datasets/UNSW-NB15/UNSW-NB15_3.csv"
unsw_nb15_4 = "../datasets/UNSW-NB15/UNSW-NB15_4.csv"

def subset(ds):
    train = pd.read_csv(unsw_nb15_tr)
    test = pd.read_csv(unsw_nb15_ts)
    ds["X_train"] = train.drop(['id', 'attack_cat', 'label'], axis = 1)
    ds["X_test"] = test.drop(['id', 'attack_cat', 'label'], axis = 1)
    ds["y_train"] = data = train['attack_cat']
    ds["y_test"] = test['attack_cat']
    ds["y_train_b"] = train['label']
    ds["y_test_b"] = test['label']
    ds["y_train_e"] = pd.get_dummies(ds["y_train"])
    ds["y_test_e"] = pd.get_dummies(ds["y_test"])
    ds["y_train_be"] = pd.get_dummies(ds["y_train_b"])
    ds["y_test_be"] = pd.get_dummies(ds["y_test_b"])
    ds["y_train_f"] = pd.factorize(train['attack_cat'])[0]
    ds["y_test_f"] = pd.factorize(test['attack_cat'])[0]
    ds["num_keys"] = ds["X_train"].iloc[:, :-2].select_dtypes(exclude=['object']).keys()
    ds["cat_keys"] = ds["X_train"].iloc[:, :-2].select_dtypes(include=['object']).keys()

def all(ds):
    cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
            'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
            'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
            'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']
    df1 = pd.read_csv(unsw_nb15_1, header=None, names = cols)
    # print (df1.head)
    df2 = pd.read_csv(unsw_nb15_2, header=None, names = cols)
    df3 = pd.read_csv(unsw_nb15_3, header=None, names = cols)
    df4 = pd.read_csv(unsw_nb15_4, header=None, names = cols)
    train = df1.append(df2).append(df3)
    test = df4
    ds["X_train"] = train.drop(['attack_cat', 'label'], axis = 1)
    ds["X_test"] = test.drop(['attack_cat', 'label'], axis = 1)
    ds["X_train"].fillna(0, inplace=True)
    ds["X_test"].fillna(0, inplace=True)
    ds["y_train"] = data = train['attack_cat']
    ds["y_test"] = test['attack_cat']
    ds["y_train_b"] = train['label']
    ds["y_test_b"] = test['label']
    ds["y_train_e"] = pd.get_dummies(ds["y_train"])
    ds["y_test_e"] = pd.get_dummies(ds["y_test"])
    ds["y_train_f"] = pd.factorize(train['attack_cat'])[0]
    ds["y_test_f"] = pd.factorize(test['attack_cat'])[0]
    ds["num_keys"] = ds["X_train"].iloc[:, :-2].select_dtypes(exclude=['object']).keys()
    print(ds["num_keys"])
    ds["cat_keys"] = ds["X_train"].iloc[:, :-2].select_dtypes(include=['object']).keys()
    print(ds["cat_keys"])