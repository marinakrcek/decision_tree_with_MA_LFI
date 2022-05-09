import os

import joblib
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
import sklearn
from reader import read_file, FileFormat


def CART_scikitlearn_decision_tree(X, Y, **kwargs):
    # function that trains the model based on sent data
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(X, Y)
    # tree.plot_tree(clf)
    # text_tree = tree.export_text(clf, feature_names=feature_names)
    # print(text_tree)
    return clf


def C50_R_decision_tree_C50(X, Y, x_column_names, rules=False):
    # imports here so no need to install these as we do not use them anyway
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects.vectors as rvectors
    import pandas as pd
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2
    # utils = rpackages.importr('utils')
    # packnames = ('C50')
    # names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    # if len('C50') > 0:
    #     utils.install_packages(StrVector(names_to_install))

    rpackages.importr('C50')
    C5_0 = rpy2.robjects.r('C5.0')
    with localconverter(rpy2.robjects.default_converter + pandas2ri.converter):
        xdf = rpy2.robjects.conversion.py2rpy(pd.DataFrame(X, columns=x_column_names))
    return C5_0(xdf, rvectors.FactorVector(Y), rules=rules)


def get_balanced_data(X, Y, class_names):
    from collections import Counter
    yC = Counter(Y)
    min_class_name = min(yC, key=yC.get)

    all_data_indexes = np.array([])
    for name in class_names:
        indexes = np.where(Y == name)[0]
        if name == min_class_name:
            all_data_indexes = np.concatenate([all_data_indexes, indexes])
            continue
        all_data_indexes = np.concatenate([all_data_indexes, np.random.choice(indexes, yC[min_class_name])])
    np.random.shuffle(all_data_indexes)
    return X[all_data_indexes.astype(int), :], Y[all_data_indexes.astype(int)]


def transform_categorical_data(data_array, decoder=None):
    data_array = np.concatenate(data_array)
    from sklearn.preprocessing import LabelEncoder
    if decoder is None:
        le = LabelEncoder()
        data_array = le.fit_transform(data_array)
        return data_array.reshape((len(data_array), 1)), le
    data_array = decoder.transform(data_array)
    return data_array.reshape((len(data_array), 1))


def resolve_parameter_bounds(bounds_final, param_bounds_file):
    if bounds_final == {}:
        bounds_final = param_bounds_file
        return bounds_final

    for param in bounds_final:
        vals = param_bounds_file[param]
        if vals[2] != bounds_final[param][2]:
            raise ValueError(f"Step is different in different logs, previous log {bounds_final[param][2]}, current log {vals[2]}")
        bounds_final[param][0] = min(vals[0], bounds_final[param][0])
        bounds_final[param][1] = max(vals[1], bounds_final[param][1])
    return bounds_final


def check_class_names(class_names_final, class_names):
    if not class_names_final:
        class_names_final = class_names
        return class_names_final

    if class_names_final != class_names:
        raise ValueError(f"Class names are not the same in the log files, please use log files "
                         f"with the same classes defined. "
                         f"From first log file: {class_names_final}, "
                         f"from current log file: {class_names}")
    return class_names_final


if __name__ == '__main__':
    root = "./train_dataset/"
    files = [file for file in os.listdir(root) if file.endswith('.json')]
    keywords = ['IC1_data_',]

    # Setting flags for learning data
    # there are 4 combinations: x, y, d, pw, i or just d, pw, i
    # and then both of those with product and test info
    no_location_info_for_learning = False
    use_target_info_data = False

    # set other parameters for decision tree
    balance_data_flags = [True, False]  # [True, False]
    criterions = ['gini', 'entropy']
    splitters = ['best', 'random']
    min_samples_splits = np.arange(2, 44, 4)
    class_weights = [None, 'balanced']
    # Minimal Cost-Complexity Pruning
    ccp_alphas = np.arange(0, 0.021, 0.005)

    X = []
    Y = np.array([])
    products = []
    tests = []
    bounds_final = {}
    class_names_final = set()
    for file in files:
        for keyword in keywords:
            if keyword not in file:
                continue
            x, y, feature_names, param_bounds, class_names, product, test = read_file(os.path.join(root, file), FileFormat.Original)
            X += [x]
            products += [np.repeat(product, len(x))]
            tests += [np.repeat(test, len(x))]
            Y = np.concatenate([Y, y])
            bounds_final = resolve_parameter_bounds(bounds_final, param_bounds)
            class_names_final = check_class_names(class_names_final, class_names)

    X_orig = np.concatenate(X)
    if no_location_info_for_learning:
        X_orig = X_orig[:, 2:]  # location is in first two columns, so remove that info
        feature_names = feature_names[2:]
    if use_target_info_data:
        products, products_decoder = transform_categorical_data(products)
        tests, tests_decoder = transform_categorical_data(tests)
        X_orig = np.append(X_orig, products, 1)
        X_orig = np.append(X_orig, tests, 1)
        feature_names += ['product', 'test']
    Y_orig = np.array(Y)
    class_conv = dict(zip(class_names_final, list(range(0, len(class_names_final)))))

    folder = os.path.join(root, "..", 'models_without_location_parameters_from_random_search')
    if not os.path.isdir(folder):
        try:
            mode = 0o666
            os.mkdir(folder,mode)
        except OSError as error:
            print(error)

    scores = {}
    for balanced_data_flag in balance_data_flags:
        if balanced_data_flag:
            X, Y = get_balanced_data(X_orig, Y_orig, class_names_final)
        else:
            X, Y = X_orig, Y_orig
        Y = np.array([class_conv[label] for label in Y])
        for criterion in criterions:
            for splitter in splitters:
                for class_weight in class_weights:
                    for min_samples_split in min_samples_splits:
                        for ccp_alpha in ccp_alphas:
                            for i in range(10):
                                X, Y = sklearn.utils.shuffle(X, Y)
                                clf = CART_scikitlearn_decision_tree(X, Y, criterion=criterion, splitter=splitter,
                                                                     min_samples_split=min_samples_split,
                                                                     class_weight=class_weight, ccp_alpha=ccp_alpha)
                                model_name = f'CART_decision_tree_criterion-{criterion}_' \
                                             f'splitter-{splitter}_min_samples_split-{str(min_samples_split)}_' \
                                             f'class_weight-{class_weight}_ccp_alpha-{str(ccp_alpha)}_' \
                                             f'balanced_data-{balanced_data_flag}_{i}'
                                # import graphviz
                                # # dot_data = tree.export_graphviz(clf, out_file=None)
                                # # graph = graphviz.Source(dot_data)
                                # dot_data = sklearn.tree.export_graphviz(clf, out_file=None, feature_names=['x', 'y', 'i', 'pw', 'd'], class_names=['Pass', 'Mute', 'Fail', 'changing'],
                                #                                 filled=True, rounded=True, special_characters=True)
                                # graph = graphviz.Source(dot_data)
                                # graph.render("lfi_campaign")
                                if use_target_info_data:
                                    joblib.dump(
                                        (clf, feature_names, bounds_final, class_conv, products_decoder, tests_decoder),
                                        os.path.join(folder, model_name + '.joblib'))
                                else:
                                    joblib.dump((clf, feature_names, bounds_final, class_conv),
                                                os.path.join(folder, model_name + '.joblib'))

    # ###### main with C50 R decition tree #######
    # clf = C50_R_decision_tree_C50(X, Y, x_column_names=feature_names, rules=False)
    # # print(clf[13]) # 13 is tree
    # print(rpy2.robjects.r('summary')(clf))
    #
    # #### predictions ####
    # with localconverter(rpy2.robjects.default_converter + pandas2ri.converter):
    #     xdf = rpy2.robjects.conversion.py2rpy(pd.DataFrame(X, columns=feature_names))
    # predict = rpy2.robjects.r('predict')
    # predictions = np.array(predict(clf, newdata=xdf))
    # print('score:', sum(predictions == np.array(rvectors.FactorVector(list(map(str, Y))))) / len(predictions))
    # ## not sure how to store: not much documentation on that
