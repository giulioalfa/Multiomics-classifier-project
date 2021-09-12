import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


def RandomForest_Classifier(X_train, X_test, y_train, y_test, dset):
    clf = RandomForestClassifier()
    param_grid = {'n_estimators' : [100, 200, 250]}
    scorer = make_scorer(accuracy_score)
    gridsearch = GridSearchCV(clf, param_grid, scoring=scorer, cv=5)
    gridsearch.fit(X_train, y_train)
    print(f"Best parameters - n_estimators: {gridsearch.best_params_['n_estimators']}")
    model = gridsearch.best_estimator_
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    if dset == 'LuadLusc100' or dset == 'ROSMAP':
        auc = roc_auc_score(y_test, y_pred)
        print(f'Accuracy: {acc} | F1: {f1} | AUC: {auc}')
    else:
        auc = None
        print(f'Accuracy: {acc} | F1: {f1}')
    return acc, f1, auc


def SVC_Classifier(X_train, X_test, y_train, y_test, dset):
    clf = SVC()
    param_grid = {'C' : [1, 2, 3],
                'kernel' : ['linear', 'rbf', 'sigmoid'],
                'gamma' : ['scale', 'auto']}
    scorer = make_scorer(accuracy_score)
    gridsearch = GridSearchCV(clf, param_grid, scoring=scorer, cv=5)
    gridsearch.fit(X_train, y_train)
    print(f"Best parameters - loss: {gridsearch.best_params_['C']}, penalty: {gridsearch.best_params_['kernel']}, gamma: {gridsearch.best_params_['gamma']}")
    clf = gridsearch.best_estimator_
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    if dset == 'LuadLusc100' or dset == 'ROSMAP':
        auc = roc_auc_score(y_test, y_pred)
        print(f'Accuracy: {acc} | F1: {f1} | AUC: {auc}')
    else:
        auc = None
        print(f'Accuracy: {acc} | F1: {f1}')
    return acc, f1, auc


def svm(dset, omics_name, omics, train, test, X_train, X_test, y_train, y_test):
    print("*******EARLY AGGREGATION RESULTS*******")
    _, _, _ = SVC_Classifier(X_train, X_test, y_train.values.ravel(), y_test.values.ravel(), dset)
    print('*******SINGLE OMIC RESULTS*******')
    for om in omics:
        print(omics_name[om])
        _, _, _ = SVC_Classifier(train[om-1], test[om-1], y_train.values.ravel(), y_test.values.ravel(), dset)


def rf(dset, omics_name, omics, train, test, X_train, X_test, y_train, y_test):
    omics_name = {1: 'mRNA', 2: 'meth', 3: 'miRNA'}
    omics_values = list(omics_name.values())
    omics_values.append('early')
    print("*******EARLY AGGREGATION RESULTS*******")
    # X_train = np.array(np.concatenate((train[0], train[1], train[2]), axis=1))
    # X_test = np.array(np.concatenate((test[0], test[1], test[2]), axis=1))
    _, _, _ = RandomForest_Classifier(X_train, X_test, y_train.values.ravel(), y_test.values.ravel(), dset)
    print('*******SINGLE OMIC RESULTS*******')
    for om in omics:
        print(omics_name[om])
        _, _, _ = RandomForest_Classifier(train[om-1], test[om-1], y_train.values.ravel(), y_test.values.ravel(), dset)


def apply_KNN(X_train, y_train, X_test, y_test, n_neighbors, dset):
    best_acc = 0
    best_k = -1
    best_i = -1
    valid_accuracy = np.empty(len(n_neighbors))
    valid_f1 = np.empty(len(n_neighbors))
    valid_auc = np.empty(len(n_neighbors))
    #print("KNN: training accuracy variation respect to k-neighbors")
    for i,k in enumerate(n_neighbors):
        clf = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        clf.fit(X_train, y_train.values.ravel())
        y_val_pred = clf.predict(X_test)
        valid_accuracy[i] = accuracy_score(y_test, y_val_pred)
        if valid_accuracy[i] > best_acc:
            best_i = i
            best_k = k
        if dset == 'LuadLusc100' or dset == 'ROSMAP':
            valid_auc[i] = roc_auc_score(y_test, y_val_pred)
            valid_f1[i] = f1_score(y_test, y_val_pred)
        else:
            valid_auc[i] = 0
            valid_f1[i] = f1_score(y_test,y_val_pred, average="weighted")
    #print("Test accuracy with n_neighbors", k, valid_accuracy[i])
    #print(f"Best accuracy : {best_acc}   and best K value : {best_k}  and best f1: {best_f1}.  and best auc: {best_auc}")
    print(f"Best K value : {best_k}")
    if dset == 'LuadLusc100' or dset == 'ROSMAP':
        print(f'Accuracy: {valid_accuracy[best_i]} | F1: {valid_f1[best_i]} | AUC: {valid_auc[best_i]}')
    else:
        print(f'Accuracy: {valid_accuracy[best_i]} | F1: {valid_f1[best_i]}')


def knn(dset, omics_name, omics, train, test, X_train, X_test, y_train, y_test):
    n_neighbors = [1, 3, 5, 7]
    print('*******SINGLE OMIC RESULTS*******')
    for om in omics:
        print(omics_name[om])
        apply_KNN(train[om-1], y_train, test[om-1], y_test, n_neighbors, dset)
    print("*******EARLY AGGREGATION RESULTS*******")
    apply_KNN(X_train, y_train, X_test, y_test, n_neighbors, dset)


def print_score(dset, ground_truth, preds):
    if dset == 'LuadLusc100' or dset == 'ROSMAP':
        print("Accuracy: {:.4f} | F1: {:.4f} | AUC: {:.4f}".format(accuracy_score(ground_truth, preds), f1_score(ground_truth, preds),
                                                               roc_auc_score(ground_truth, preds)))
    else:
        print("Accuracy: {:.4f} | F1: {:.4f}".format(accuracy_score(ground_truth, preds), f1_score(ground_truth, preds, average="weighted")))


def apply_RIDGE(X_train, y_train, X_test, y_test, dset):
  clf = RidgeClassifier()
  param_grid = {'alpha': [0.01, 0.1, 0.5, 1]}
  scorer = make_scorer(accuracy_score)
  gridsearch = GridSearchCV(clf, param_grid, scoring=scorer, cv=5)
  gridsearch.fit(X_train, y_train.values.ravel())
  print(f"Best parameters - loss: {gridsearch.best_params_['alpha']}")
  clf = gridsearch.best_estimator_
  # clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print_score(dset, y_test, y_pred)


def ridge(dset, omics_name, omics, train, test, X_train, X_test, y_train, y_test):
    print('*******SINGLE OMIC RESULTS*******')
    for om in omics:
        print(omics_name[om])
        apply_RIDGE(train[om-1], y_train, test[om-1], y_test, dset)
    print("*******EARLY AGGREGATION RESULTS*******")
    apply_RIDGE(X_train, y_train, X_test, y_test, dset)


def apply_LASSO(X_train, y_train, X_test, y_test, dset):
    clf = LogisticRegression(penalty='l1', solver='liblinear')
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    scorer = make_scorer(accuracy_score)
    gridsearch = GridSearchCV(clf, param_grid, scoring=scorer, cv=5)
    gridsearch.fit(X_train, y_train.values.ravel())
    print(f"Best parameters - loss: {gridsearch.best_params_['C']}")
    clf = gridsearch.best_estimator_
    # clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_score(dset, y_test, y_pred)


def lasso(dset, omics_name, omics, train, test, X_train, X_test, y_train, y_test):
    print('*******SINGLE OMIC RESULTS*******')
    for om in omics:
        print(omics_name[om])
        apply_LASSO(train[om-1], y_train, test[om-1], y_test, dset)
    print("*******EARLY AGGREGATION RESULTS*******")
    apply_LASSO(X_train, y_train, X_test, y_test, dset)


def main(clf):
    datasets = ['LuadLusc100', '5000samples', 'BRCA', 'ROSMAP']
    dir = '../Data/'
    omics = [1, 2, 3]
    omics_name = {1: 'mRNA', 2: 'meth', 3: 'miRNA'}
    for dset in datasets:
        curr_dir = dir + dset + '/'
        print(f"*******{dset}*******")
        train = {}
        test = {}
        for om in omics:
            train[om-1] = pd.read_csv(curr_dir + str(om) + '_tr.csv', header=None)
            test[om-1] = pd.read_csv(curr_dir + str(om) + '_te.csv', header=None)
        y_train = pd.read_csv(curr_dir + 'labels_tr.csv', header=None)
        y_test = pd.read_csv(curr_dir + 'labels_te.csv', header=None)
        X_train = np.array(np.concatenate((train[0], train[1], train[2]), axis=1))
        X_test = np.array(np.concatenate((test[0], test[1], test[2]), axis=1))
        clf(dset, omics_name, omics, train, test, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    print("Select an alghorithm with corresponding number:")
    print("1 - SVM\n2 - Random Forest\n3 - KNN\n4 - Ridge\n5 - Lasso\n")
    c = int(input('Number: '))
    if c == 1:
        main(svm)
    elif c == 2:
        main(rf)
    elif c == 3:
        main(knn)
    elif c == 4:
        main(ridge)
    elif c == 5:
        main(lasso)
    else:
        print("Error")