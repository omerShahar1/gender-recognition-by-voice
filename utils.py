import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

age_label2int = {
    "teens": 0,
    "twenties": 1,
    "thirties": 2,
    "fourties": 3,
    "fifties": 4,
    "sixties": 5,
    "seventies": 6,
    "eighties": 6,
    "nineties": 6
}


def load_data(vector_length=193):
    """A function to load age recognition dataset from `data` folder
    After the second run, this will load from results/features.npy and results/labels.npy files
    as it is much faster!"""
    # make sure results folder exists
    if not os.path.isdir("results"):
        os.mkdir("results")
    # if features & labels already loaded individually and bundled, load them from there instead
    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        x = np.load("results/features.npy")
        y = np.load("results/labels.npy")
        return x, y

    df = pd.read_csv("balanced_all.csv")  # read dataframe
    n_samples = len(df)  # get total samples and amount of samples for all groups
    n_0_samples = len(df[df['age'] == 'teens'])
    n_1_samples = len(df[df['age'] == 'twenties'])
    n_2_samples = len(df[df['age'] == 'thirties'])
    n_3_samples = len(df[df['age'] == 'fourties'])
    n_4_samples = len(df[df['age'] == 'fifties'])
    n_5_samples = len(df[df['age'] == 'sixties'])
    n_6_samples = len(df[df['age'] == 'seventies']) + len(df[df['age'] == 'eighties']) + len(df[df['age'] == 'nineties'])

    print("Total samples:", n_samples)
    print("10-19 samples:", n_0_samples)
    print("20-29 samples:", n_1_samples)
    print("30-39 samples:", n_2_samples)
    print("40-49 samples:", n_3_samples)
    print("50-59 samples:", n_4_samples)
    print("60-69 samples:", n_5_samples)
    print("70 or older samples:", n_6_samples)


    x = np.zeros((n_samples, vector_length))  # initialize an empty array for all audio features
    y = np.zeros((n_samples, 1))  # initialize an empty array for all audio labels (age)
    for i, (filename, age) in tqdm(enumerate(zip(df['filename'], df['age'])), "Loading data", total=n_samples):
        if age in age_label2int:
            features = np.load("data/" + filename)
            x[i] = features
            y[i] = age_label2int[age]

    # save the audio features and labels into files,
    # so we won't load each one of them next run
    np.save("results/features", x)
    np.save("results/labels", y)
    return x, y


def compute_weight(full_label_dict):
    weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.ravel(full_label_dict, order='C')),
                                         y=np.ravel(full_label_dict, order='C'))

    return dict(zip(np.unique(full_label_dict), weights))


def split_data(x, y, test_size=0.1, valid_size=0.1):
    # split training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    # split training set and validation set
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size, random_state=42)

    # return a dictionary of values
    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "x_test": x_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }


def create_model(vector_length=56):
    """5 hidden dense layers from 256 units to 64."""
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation="Softmax"))
    model.compile(loss="SparseCategoricalCrossentropy", metrics=["accuracy"], optimizer="adam")

    # print summary of the model
    model.summary()
    return model