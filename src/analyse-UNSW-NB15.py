import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import train_predict as tp
import visuals as vs

do_data_preprocessing = False
do_pca_analysis = True
do_supervised_learning = True
do_lstm_rnn = False

if do_data_preprocessing:
    #######################################################################################################################
    # Load and Explore Data
    #######################################################################################################################

    # Load Data
    training_data = pd.read_csv("../datasets/UNSW-NB15/UNSW_NB15_testing-set.csv")
    testing_data = pd.read_csv("../datasets/UNSW-NB15/UNSW_NB15_training-set.csv")
    # data = pd.concat([training_data, testing_data])

    # Explore Data
    n_training_records = len(training_data)
    n_testing_records = len(testing_data)
    normal_training_data_count = training_data["label"].where(training_data["label"] == 0).count()
    normal_testing_data_count = testing_data["label"].where(testing_data["label"] == 0).count()
    attack_training_data_count = training_data["label"].where(training_data["label"] == 1).count()
    attack_testing_data_count = testing_data["label"].where(testing_data["label"] == 1).count()
    attack_training_data_percentage = (float(attack_training_data_count)/float(n_training_records)) * 100
    attack_testing_data_percentage = (float(attack_testing_data_count)/float(n_testing_records)) * 100


    #-------------------------
    # Address training and testing dataset missmatch

    training_data.loc[training_data.state == 'no', 'state'] = "FIN"
    training_data.loc[training_data.state == 'ECO', 'state'] = "FIN"
    training_data.loc[training_data.state == 'PAR', 'state'] = "FIN"
    training_data.loc[training_data.state == 'URN', 'state'] = "FIN"

    testing_data.loc[testing_data.state == 'ACC', 'state'] = "FIN"
    testing_data.loc[testing_data.state == 'CLO', 'state'] = "FIN"

    training_data.loc[training_data.proto == 'icmp', 'proto'] = "tcp"
    training_data.loc[training_data.proto == 'rtp', 'proto'] = "tcp"



    #--------------------------




    # Print Data Exploration results
    print ("Total number of training records: {}".format(n_training_records))
    print ("Total number of testing records: {}".format(n_testing_records))
    print ("Total number of normal training records: {}".format(normal_training_data_count))
    print ("Total number of normal testing records: {}".format(normal_testing_data_count))
    print ("Total number of attack training records: {}".format(attack_training_data_count))
    print ("Total number of attack testing records: {}".format(attack_testing_data_count))
    print ("Percentage of attack traffic in training: {:.2f}%".format(attack_training_data_percentage))
    print ("Percentage of attack traffic in testing: {:.2f}%".format(attack_testing_data_percentage))

    # Split the data into features and target labels
    tr_attack_labels = pd.DataFrame(data = training_data['attack_cat'])
    te_attack_labels = pd.DataFrame(data = testing_data['attack_cat'])
    tr_binary_labels = pd.DataFrame(data = training_data['label'])
    te_binary_labels = pd.DataFrame(data = testing_data['label'])
    tr_features = training_data.drop(['id', 'attack_cat', 'label'], axis = 1)
    te_features = testing_data.drop(['id', 'attack_cat', 'label'], axis = 1)

    #Split the features into numerical and categorical features
    numerical_features = tr_features.iloc[:, :-2].select_dtypes(exclude=['object'])
    categorical_features = tr_features.iloc[:, :-2].select_dtypes(include=['object'])

    # More data exploration
    #TODO: Read feature file and display description
    #TODO: Display unique outliers with count
    # print ("feature,datatype,mean,median,mode,min,max,25%,50%,75%,skew,std,var")
    # for feature in numerical_features.keys():
    #     s = numerical_features[feature]
    #     print("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}"
    #           .format(feature, s.dtype, s.mean(), s.median(), s.min(), s.max(), s.quantile(.25)
    #                   , s.quantile(.5),s.quantile(.75),s.skew(),s.std(), s.var()))
    #
    # for feature in categorical_features.keys():
    #     print(feature)
    # print(numerical_features.corr())




    #######################################################################################################################
    # Preprocess and Prepare Data
    #######################################################################################################################

    # Apply Log Transformation to features. Display pairwise correlation before and after transformation
    # sns.heatmap(tr_features[numerical_features.keys()].corr())
    # plt.show()
    tr_features_log_trans = pd.DataFrame(data = tr_features)
    tr_features_log_trans[numerical_features.keys()] = tr_features[numerical_features.keys()].apply(lambda x: np.log(x + 1))
    # sns.heatmap(tr_features_log_trans[numerical_features.keys()].corr())
    # plt.show()
    tr_features_log_trans.to_csv("../interim-files/tr_features_log_trans.csv", index=False)


    # sns.heatmap(te_features[numerical_features.keys()].corr())
    # plt.show()
    te_features_log_trans = pd.DataFrame(data = te_features)
    te_features_log_trans[numerical_features.keys()] = te_features[numerical_features.keys()].apply(lambda x: np.log(x + 1))
    # sns.heatmap(te_features_log_trans[numerical_features.keys()].corr())
    # plt.show()
    te_features_log_trans.to_csv("../interim-files/te_features_log_trans.csv", index=False)


    # Normalize Numerical Features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    tr_features_log_minmax_trans = pd.DataFrame(data = tr_features_log_trans)
    tr_features_log_minmax_trans[numerical_features.keys()] = scaler.fit_transform(tr_features_log_trans[numerical_features.keys()])
    tr_features_log_minmax_trans.to_csv("../interim-files/tr_features_log_minmax_trans.csv", index=False)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    te_features_log_minmax_trans = pd.DataFrame(data = te_features_log_trans)
    te_features_log_minmax_trans[numerical_features.keys()] = scaler.fit_transform(te_features_log_trans[numerical_features.keys()])
    te_features_log_minmax_trans.to_csv("../interim-files/te_features_log_minmax_trans.csv", index=False)

    # One-hot encode categorical features
    tr_features_log_minmax_onehot_trans = pd.get_dummies(tr_features_log_minmax_trans)
    tr_features_log_minmax_onehot_trans.to_csv("../interim-files/tr_features_log_minmax_onehot_trans.csv", index=False)
    tr_features_count = list(tr_features_log_minmax_onehot_trans.columns)
    print ("{} total features after one-hot encoding in training set.".format(len(tr_features_count)))

    te_features_log_minmax_onehot_trans = pd.get_dummies(te_features_log_minmax_trans)
    te_features_log_minmax_onehot_trans.to_csv("../interim-files/te_features_log_minmax_onehot_trans.csv", index=False)
    te_features_count = list(te_features_log_minmax_onehot_trans.columns)
    print ("{} total features after one-hot encoding in testing set.".format(len(te_features_count)))


    # One hot encode attack labels
    tr_encoded_labels = pd.get_dummies(tr_attack_labels)
    tr_encoded_labels.to_csv("../interim-files/tr_encoded_labels.csv", index=False)
    tr_encoded_labels_count = list(tr_encoded_labels.columns)
    print ("{} total label columns after one-hot encoding in training set.".format(len(tr_encoded_labels_count)))

    te_encoded_labels = pd.get_dummies(te_attack_labels)
    te_encoded_labels.to_csv("../interim-files/te_encoded_labels.csv", index=False)
    te_encoded_labels_count = list(te_encoded_labels.columns)
    print ("{} total label columns after one-hot encoding in testing set.".format(len(te_encoded_labels_count)))

    tr_binary_labels.to_csv("../interim-files/tr_binary_labels.csv", index=False)
    te_binary_labels.to_csv("../interim-files/te_binary_labels.csv", index=False)
    tr_attack_labels.to_csv("../interim-files/tr_attack_labels.csv", index=False)
    te_attack_labels.to_csv("../interim-files/te_attack_labels.csv", index=False)


    X_train = pd.DataFrame(data=tr_features_log_minmax_onehot_trans)
    X_test = pd.DataFrame(data=te_features_log_minmax_onehot_trans)
    ye_train = pd.DataFrame(data=tr_encoded_labels)
    ye_test  = pd.DataFrame(data=te_encoded_labels)
    yb_train = pd.DataFrame(data=tr_binary_labels)
    yb_test  = pd.DataFrame(data=te_binary_labels)
    y_train = pd.DataFrame(data=tr_attack_labels)
    y_test = pd.DataFrame(data=te_attack_labels)

else:

    X_train = pd.read_csv("../interim-files/tr_features_log_minmax_onehot_trans.csv")
    X_test = pd.read_csv("../interim-files/te_features_log_minmax_onehot_trans.csv")
    ye_train = pd.read_csv("../interim-files/tr_encoded_labels.csv")
    ye_test  = pd.read_csv("../interim-files/te_encoded_labels.csv")
    yb_train = pd.read_csv("../interim-files/tr_binary_labels.csv")
    yb_test  = pd.read_csv("../interim-files/te_binary_labels.csv")
    y_train = pd.read_csv("../interim-files/tr_attack_labels.csv")
    y_test = pd.read_csv("../interim-files/te_attack_labels.csv")

print ("X_train set has {} shape.".format(X_train.shape))
print ("ye_train set has {} shape.".format(ye_train.shape))
print ("yb_train set has {} shape.".format(yb_train.shape))
print ("X_test set has {} shape.".format(X_test.shape))
print ("ye_test set has {} shape.".format(ye_test.shape))
print ("yb_test set has {} shape.".format(yb_test.shape))

#######################################################################################################################
# Dimensionality Reduction
#######################################################################################################################


if do_pca_analysis:
    from sklearn.decomposition import PCA

    pca_2c = PCA(n_components=100, random_state=0)
    pca_2c.fit(X_train)
    Xr_train = pca_2c.transform(X_train)
    Xr_test = pca_2c.transform(X_test)



    # pca = PCA(n_components=2, random_state=0)
    # pca.fit(good_data)
    #
    # # TODO: Transform the good data using the PCA fit above
    # reduced_data = pca.transform(good_data)
    #
    # # TODO: Transform log_samples using the PCA fit above
    # pca_samples = pca.transform(log_samples)
    #
    # # Create a DataFrame for the reduced data
    # reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
    #
    # vs.biplot(good_data, reduced_data, pca)

#######################################################################################################################
# Create a Naive Predictor and Supervised Learning Models
#######################################################################################################################

if do_supervised_learning:
    TP = np.sum(yb_train)
    FP = yb_train.count() - TP
    TN = 0
    FN = 0


    # Calculate accuracy, precision and recall
    accuracy = float(TP)/float(yb_train.count())
    recall = float(TP)/(float(TP) + float(FN))
    precision = float(TP)/(float(TP) + float(FP))

    beta = 0.001
    fscore = (1 + beta**2) * (float(precision) * float(recall)) / ((beta**2 * float(precision)) + float(recall))

    # Print the results
    print ("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


    # TODO: Import the three supervised learning models from sklearn
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import linear_model
    # from sklearn import svm
    from sklearn import neighbors
    from sklearn import linear_model
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm

    # TODO: Initialize the three models
    clf_A = DecisionTreeClassifier(random_state = 0)
    clf_B = linear_model.SGDClassifier(random_state = 0)
    # clf_C = svm.SVC(random_state = 0)
    clf_D = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform')
    clf_E = linear_model.LogisticRegression(random_state = 0)
    # clf_F = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf_G = RandomForestClassifier(max_depth=2, random_state=0)

    # TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
    # HINT: samples_100 is the entire training set i.e. len(y_train)
    # HINT: samples_10 is 10% of samples_100
    # HINT: samples_1 is 1% of samples_100
    samples_100 = int(len(y_train))
    samples_10 = int(len(y_train)/10)
    samples_1 = int(len(y_train)/100)

    # Collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_D, clf_E, clf_G]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_100]):
            print("sample size: ", samples)
            results[clf_name][i] = \
            tp.train_predict(clf, samples, Xr_train, y_train, Xr_test, y_test)

    # Run metrics visualization for the three supervised learning models chosen
    vs.evaluate(results, accuracy, fscore)

if do_lstm_rnn:

    from keras.models import Sequential
    from keras.layers import LSTM, Dense, TimeDistributed, AveragePooling1D, Flatten

    X_train_3d = X_train.as_matrix().reshape((175341,188,1))
    X_test_3d = X_test.as_matrix().reshape((82332,188,1))
    ye_train_3d = ye_train.as_matrix().reshape((175341, 1, 10, 1))
    ye_test_3d = ye_test.as_matrix().reshape((82332, 1, 10, 1))
    #
    #
    #
    # model = Sequential()
    # model.add(LSTM(1, return_sequences=True, input_shape=(82332,188)))
    # # model.add(LSTM(188))
    # # model.add(TimeDistributed(Dense(1)))
    # # model.add(AveragePooling1D())
    #
    # model.add(Flatten())
    # model.add(Dense(10, activation="softmax"))
    # model.compile(optimizer='adam', loss='mse')
    #
    # model.fit(X_train_3d, ye_train.as_matrix(),
    #           epochs=20,
    #           batch_size=128)
    # score = model.evaluate(X_test_3d, ye_test.as_matrix(), batch_size=128)

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD
    from keras.utils import np_utils

    model = Sequential()
    model.add(LSTM(1, activation='relu', return_sequences=True, input_shape=(188,1)))
    model.add(Dropout(.2))
    model.add(LSTM(1, activation='relu'))
    # model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    # Compiling the model
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    from keras.callbacks import ModelCheckpoint

    # train the model
    checkpointer = ModelCheckpoint(filepath='MLP.weights.best.hdf5', verbose=1,
                                   save_best_only=True)
    hist = model.fit(x_train, y_train, batch_size=32, epochs=20,
              validation_data=(x_valid, y_valid), callbacks=[checkpointer],
              verbose=2, shuffle=True)

    model.fit(X_train_3d, ye_train, epochs=2, batch_size=100, verbose=0)

    score = model.evaluate(X_train_3d, ye_train)
    print("\n Training Accuracy:", score)
    score = model.evaluate(X_test_3d, ye_test)
    print("\n Testing Accuracy:", score)