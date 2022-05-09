from enum import Enum
import numpy as np
import json
import pandas as pd


class FileFormat(Enum):
    Anonymized = 1
    Original = 2
    SimpleAnonymized = 3
    OriginalCSV = 4


def read_original_json_log(file_name):
    with open(file_name, mode='r') as fp:
        data = json.load(fp)
    metadata = data['metadata']
    status = metadata['STATUS']
    header, param_bounds = data['header'], metadata['parameter_bounds']
    param_names = list(param_bounds.keys())
    indexes = [header.index(name) for name in param_names]
    assert (len(indexes) == len(param_names))
    indexes.append(header.index(status))
    results = np.array(data['results'])
    df = pd.DataFrame(results[:, indexes], columns=param_names + [status])
    df = pd.DataFrame(df.groupby(param_names)[status].apply(lambda x: list(x)[0] if len(set(x)) == 1 else 'changing')).reset_index()
    Y = np.array(df[status])
    X = np.array(df[param_names].apply(pd.to_numeric))
    return X, Y, param_names, param_bounds, set(Y), metadata['Product'], metadata['test']


def read_anonymized_json_full(file_name):
    with open(file_name, mode='r') as fp:
        data = json.load(fp)
    metadata = data['metadata']
    param_bounds = metadata['parameter_bounds']
    param_names = list(param_bounds.keys())
    header = data['header']
    indexes = [header.index(name) for name in param_names]
    assert (len(indexes) == len(param_names))
    status = metadata['STATUS']
    indexes.append(header.index(status))
    results = np.array(data['results'])
    df = pd.DataFrame(results[:, indexes], columns=param_names + [status])
    df = pd.DataFrame(df.groupby(param_names)[status].apply(lambda x: list(x)[0] if len(set(x)) == 1 else 'changing')).reset_index()
    Y = np.array(df[status])
    X = np.array(df[param_names].apply(pd.to_numeric))
    return X, Y, param_names, param_bounds, set(Y), metadata['Product'], metadata['test']


def read_file(file_name, file_format):
    if file_format == FileFormat.Anonymized:
        return read_anonymized_json_full(file_name)
    elif file_format == FileFormat.Original:
        return read_original_json_log(file_name)
    else:
        raise ValueError("Wrong format for reading the log.")
