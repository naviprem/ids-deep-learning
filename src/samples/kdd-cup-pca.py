#########
# Code credits: https://github.com/gfleetwood/pca-presentation/blob/master/PCA_Presentation.ipynb
#########


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn.cross_validation as cv
from sklearn.linear_model import LogisticRegression

with open('../datasets/KDD-CUP-99/kddcup.data_10_percent.txt', 'r') as file_kdd_access:

    cols = ["duration","protocol_type","service","flag","src_bytes", "dst_bytes","land",
            "wrong_fragment","urgent","hot","num_failed_logins", "logged_in","num_compromised",
            "root_shell","su_attempted","num_root", "num_file_creations","num_shells",
            "num_access_files","num_outbound_cmds", "is_host_login","is_guest_login","count",
            "srv_count","serror_rate", "srv_serror_rate","rerror_rate","srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate","srv_diff_host_rate","dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate","dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate","dst_host_srv_rerror_rate","TARGET"]

    kdd_data = pd.read_csv(file_kdd_access, header=None, names=cols, low_memory=False)

    # print(kdd_data.head())

    print('10% of the data is {} data points and has {} columns.'.format(len(kdd_data), len(kdd_data.columns)),
          end='\n\n')
    # print('Missingness: ', end='\n\n')
    # print(kdd_data.isnull().sum())
    print('There are {} unique targets'.format(len(kdd_data.TARGET.unique())), end='\n\n')
    # print(kdd_data.TARGET.value_counts(normalize=True))

    kdd_data['BINARY_TARGET'] = kdd_data['TARGET'].map(lambda x: x if x == 'normal.' else 'abnormal.')
    print(kdd_data.BINARY_TARGET.value_counts(normalize=True))

    # print(kdd_data.dtypes)

    weird_cols = ['num_root', 'num_file_creations', 'num_shells']

    for col in weird_cols:
        print('\n' + col)
        print(list(filter(lambda x: not x[1], [(x, x.isdigit()) for x in kdd_data[col].values])))

    kdd_data.loc[kdd_data.num_root == 'tcp', 'num_root'] = kdd_data.num_root.value_counts().index[0]
    kdd_data.loc[kdd_data.num_file_creations == 'http', 'num_file_creations'] = kdd_data.num_file_creations.value_counts().index[0]
    kdd_data.loc[kdd_data.num_shells == 'SF', 'num_shells'] = kdd_data.num_shells.value_counts().index[0]

    kdd_data.loc[:, weird_cols] = kdd_data.loc[:, weird_cols].apply(pd.to_numeric)

    kdd_data.loc[kdd_data.su_attempted == 2, 'su_attempted'] = 0

    kdd_data_num_features = kdd_data.iloc[:, :-2].select_dtypes(exclude=['object'])
    kdd_data_cat_features = kdd_data.iloc[:, :-2].select_dtypes(include=['object'])

    print(kdd_data_num_features.describe())
    print(kdd_data_cat_features.describe())

    sns.heatmap(kdd_data_num_features.corr())
    plt.show()


    def column_encoding(df_num, df_cat):
        for i in range(len(df_cat.columns)):
            df_num = pd.concat([df_num, pd.get_dummies(df_cat.iloc[:, i])], axis=1)
        return df_num


    kdd_features_dummied = column_encoding(kdd_data_num_features, kdd_data_cat_features)
    len(kdd_features_dummied.columns)

    scaled_X_dummied = StandardScaler().fit_transform(kdd_features_dummied)
    pca_kdd = PCA(n_components=2)
    scaled_X_dummied_pca_2 = pca_kdd.fit_transform(scaled_X_dummied)

    normal_indices = kdd_data[kdd_data.TARGET != 'normal.'].index.values
    abnormal_indices = kdd_data[kdd_data.TARGET == 'normal.'].index.values
    plt.scatter(scaled_X_dummied_pca_2[normal_indices, 0], scaled_X_dummied_pca_2[normal_indices, 1], c='b', )
    plt.scatter(scaled_X_dummied_pca_2[abnormal_indices, 0], scaled_X_dummied_pca_2[abnormal_indices, 1], c='r')
    plt.title('Network Intrusion In 2D')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(['normal', 'abnormal'])
    plt.show()


    def kaiser_harris_criterion(cov_mat):
        e_vals, _ = np.linalg.eig(cov_mat)
        return len(e_vals[e_vals > 1])

    # Choose Subspace Dimension
    cov = np.cov(scaled_X_dummied.T)
    print (cov)
    k_harris_rec = kaiser_harris_criterion(cov)
    print('Kaiser-Harris Criterion: Use {} principal components.'.format(k_harris_rec))

    # pca_kdd.set_params(n_components=k_harris_rec)
    # scaled_X_dummied_pca_k_harris = pca_kdd.fit_transform(scaled_X_dummied)
    #
    # y_train = kdd_data.iloc[:, -1].map(lambda x: 1 if x == 'normal.' else 0)
    # stratified_divide = cv.StratifiedKFold(y_train, n_folds=10, random_state=1)
    # clf_log_reg2 = LogisticRegression()
    # clf_log_reg2_cv_score = np.mean(cv.cross_val_score(clf_log_reg2,
    #                                                    scaled_X_dummied_pca_k_harris,
    #                                                    y_train,
    #                                                    cv=stratified_divide,
    #                                                    scoring='accuracy'))
    # print(clf_log_reg2_cv_score)