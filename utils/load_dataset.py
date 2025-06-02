"""
Methods from Skrodzki et al. paper to load relevant datasets
"""

from pathlib import Path
import os
import gzip

import numpy as np
import pandas as pd


def get_mnist(path, kind="all"):
    """
    TODO Docstring for get_mnist.
    """
    path_to_data = Path(path)
    if not path_to_data.exists():
        raise Exception("mnist data was not found at {}".format(path_to_data))

    labels_path_train = os.path.join(path_to_data, 'train-labels-idx1-ubyte.gz')
    labels_path_test = os.path.join(path_to_data, 't10k-labels-idx1-ubyte.gz')
    images_path_train = os.path.join(path_to_data, 'train-images-idx3-ubyte.gz')
    images_path_test = os.path.join(path_to_data, 't10k-images-idx3-ubyte.gz')

    labels_dict = dict()
    images_dict = dict()

    if kind == 'all' or kind == 'train':
        with gzip.open(labels_path_train, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_dict["train"] = br

    if kind == 'all' or kind == 'test':
        with gzip.open(labels_path_test, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_dict["test"] = br

    if kind == 'all' or kind == 'train':
        with gzip.open(images_path_train, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_dict["train"]), 784)
            images_dict["train"] = images

    if kind == 'all' or kind == 'test':
        with gzip.open(images_path_test, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_dict["test"]), 784)
            images_dict["test"] = images

    labels = np.concatenate(list(labels_dict.values()), axis=0)
    images = np.concatenate(list(images_dict.values()), axis=0)

    return images, labels

def load_wong(data_home, labels_name=None, return_colors=False, return_numeric_labels=True):
    """
    TODO Docstring
    """

    return_labels = False
    if labels_name is not None and labels_name in ["broad", "organs"]:
        return_labels = True
    
    data_home = Path(data_home)
    print(data_home)
    if not data_home.exists():
        raise Exception("wong data was not found at {}".format(data_home))
        
    path_parsed_csv = data_home.joinpath("10k_parsed.csv")
    if not path_parsed_csv.exists():
        raise Exception("preprocess wong data using `parse_data.R` was not found at {}".format(path_parsed_csv))
    
    path_labels = data_home.joinpath("{}_colors.csv".format(labels_name))
    if not path_labels.exists():
        print("labels path not found labels will not be returned".format(path_labels))
        return_labels = False

    X = pd.read_csv(path_parsed_csv).to_numpy()

    if return_labels:
        labels_df = pd.read_csv(path_labels)
        labels = labels_df[f"{labels_name}_color"] if return_colors else labels_df[f"{labels_name}_name"]
        if return_numeric_labels:
            out_labs = np.zeros(labels.size)
            for i, l in enumerate(np.unique(labels)):
                out_labs[labels == l] = i
            labels = out_labs

    if return_labels:
        return X, labels
    else:
        return X
    

def load_celegans(data_home, return_X_y=True):
    """
    Loads C-ELEGANS data available at https://data.caltech.edu/records/1945 

    Parameters
    __________
    data_home : str, optional
        Locations of the folder where the datasets are stored.
    return_X_y: bool, optional
        If True, method only returns tuple with the data and its labels.
    """    
    import anndata as ad

    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "c_elegans")

    ad_obj = ad.read_h5ad(str(Path.joinpath(full_path, "packer2019.h5ad")))
    X = ad_obj.X

    labels_str = np.array(ad_obj.obs.cell_type)

    _, labels = np.unique(labels_str, return_inverse=True)

    print(labels.shape)

    if return_X_y:
        return X, labels
    else:
        return X