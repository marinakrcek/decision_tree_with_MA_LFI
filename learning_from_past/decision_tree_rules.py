from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn import tree
import joblib
import numpy as np


def prepare_for_arange(min_max_step_list):
    return min_max_step_list[0], min_max_step_list[1]+min_max_step_list[2], min_max_step_list[2]


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def get_simplified_rules(tree, feature_names, parameter_bounds, for_class=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)

            p1 += [(name, '<=', np.round(threshold, 3))]
            recurse(tree_.children_left[node], p1, paths)

            p2 += [(name, '>', np.round(threshold, 3))]
            recurse(tree_.children_right[node], p2, paths)
        else:
            if for_class is not None and np.argmax(tree_.value[node]) != for_class:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                return

            larger_than = {f_name: [] for f_name in feature_names}
            less_eq_than = {f_name: [] for f_name in feature_names}

            for name, sign, value in path:
                if sign == '>':
                    larger_than[name] += [value]
                else:
                    less_eq_than[name] += [value]

            path_reduced = {}
            for key, larger_than_value in larger_than.items():
                less_eq_than_value = less_eq_than[key]
                step = parameter_bounds[key][2]
                lower = parameter_bounds[key][0]
                if larger_than_value:
                    lower = np.max(larger_than_value)
                    lower = lower + (step if lower in np.arange(*prepare_for_arange(parameter_bounds[key])) else step/2)
                    assert (lower in np.arange(*prepare_for_arange(parameter_bounds[key]))), f"{lower} not in param bound for {key}, bounds: {parameter_bounds[key]}"
                upper = parameter_bounds[key][1] + step
                if less_eq_than_value:
                    upper = np.min(less_eq_than_value)
                    upper = upper + (step if upper in np.arange(*prepare_for_arange(parameter_bounds[key])) else step/2)
                    assert (upper in np.arange(*prepare_for_arange(parameter_bounds[key]))), f"{upper} not in param bound for {key}, bounds: {parameter_bounds[key]}"
                path_reduced[key] = [lower, upper, parameter_bounds[key][2]]

            path_reduced['status'] = {'class_index': np.argmax(tree_.value[node]),
                                      'nb_samples_per_class': tree_.value[node][0]}
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path_reduced]

    recurse(0, path, paths)

    # # sort by samples count
    # samples_count = [p['status']['nb_samples_per_class'][p['status']['class_index']] for p in paths]
    # ii = list(np.argsort(samples_count))
    # paths = [paths[i] for i in reversed(ii)]
    return paths


def get_bounds_from_decoder(decoder):
    """
    Return bounds for a parameter based on the decoder (for parameters such as product and test).
    :param decoder: Decoder used during training.
    :return: bounds for the parameter that is not for expanding (max has to be increased if used in np.arange)
    """
    numbers = decoder.transform(decoder.classes_)
    return [np.min(numbers), max(numbers), 1]  # not for expanding


def get_rules_for_initialization(from_file, for_class=None, product=None, test=None):
    """
    From the trained decision tree model (from_file) extracts the rules for for_class fault class.
    Usually we choose fail class, but it does not have to be.
    The function reads the joblib file with respect to the data stored inside the file.
    The number of features used for training is detected here from that information.
    Rules are reduced to only rules that are valid for the sent product and test information if that was a parameter
    used for training of the decision tree.
    :param from_file: From which trained decision tree model.
    :param for_class: For which class do you want to get the rules for.
    :param product: Send only when you know that decision tree was learned using that target information.
    :param test: As for product.
    :return:
    """
    data = joblib.load(from_file)
    if len(data) == 4:
        clf, feature_names, param_bounds, class_conv = data
    elif len(data) == 6:
        clf, feature_names, param_bounds, class_conv, products_decoder, tests_decoder = data
        param_bounds['product'] = get_bounds_from_decoder(products_decoder)
        param_bounds['test'] = get_bounds_from_decoder(tests_decoder)
        if product is None or test is None:
            raise ValueError('Since learning used product and test information, please send that information for the current campaign.')
        product = products_decoder.transform([product])[0]
        test = tests_decoder.transform([test])[0]
    else:
        raise ValueError(f'Expected 4 or 6 variables from joblib, got {len(data)}.')

    rules = get_simplified_rules(clf, feature_names=feature_names, parameter_bounds=param_bounds,
                                for_class=class_conv[for_class] if for_class is not None else None)
    if len(data) != 6:
        # no product and test information was used for training, so just send all the rules
        return rules
    # clean rules for the product that we run the experiment on
    rules = [rule for rule in rules if product in np.arange(*rule['product']) and test in np.arange(*rule['test'])]
    return rules
