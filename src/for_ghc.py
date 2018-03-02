import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV


unsw_nb15_1 = "../datasets/UNSW-NB15/UNSW-NB15_1.csv"
unsw_nb15_2 = "../datasets/UNSW-NB15/UNSW-NB15_2.csv"
unsw_nb15_3 = "../datasets/UNSW-NB15/UNSW-NB15_3.csv"
unsw_nb15_4 = "../datasets/UNSW-NB15/UNSW-NB15_4.csv"
random_state = 55

seed = np.random.seed(random_state)

class ModalData:

    def __init__(self):
        self.rs = random_state
        self.load()

    def load(self):
        cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
                'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
                'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
                'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
                'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']
        df1 = pd.read_csv(unsw_nb15_1, header=None, names=cols)
        df2 = pd.read_csv(unsw_nb15_2, header=None, names=cols)
        df3 = pd.read_csv(unsw_nb15_3, header=None, names=cols)
        df4 = pd.read_csv(unsw_nb15_4, header=None, names=cols)
        dataset = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
        # dataset = df1.append(df2).append(df3).append(df4)
        labels = pd.DataFrame(dataset['label'], columns=['label'])
        print(labels.describe())
        with_markers = pd.concat([labels, labels.index.to_series().map(lambda row: int((row + 1)/1000)), labels.index.to_series().map(lambda row: int((row + 1)%1000))], axis=1)
        with_markers.columns = ['label', 'x', 'y']
        print(with_markers.head)
        attack = with_markers[with_markers['label'] == 1]
        normal = with_markers[with_markers['label'] == 0]
        print("Total number of normal records: ", len(normal))
        print("Total number of attack records: ", len(attack))

        plt.figure(figsize=(250, 100))
        plt.scatter(x=attack.loc[:, 'x'], y=attack.loc[:, 'y'],
               facecolors='r', edgecolors='r', s=10, alpha=0.5)
        plt.scatter(x=normal.loc[:, 'x'], y=normal.loc[:, 'y'],
                    facecolors='b', edgecolors='b', s=10, alpha=0.5)
        plt.savefig("labels.png")
        plt.show()




data = ModalData()