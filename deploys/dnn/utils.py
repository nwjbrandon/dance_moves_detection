import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(70, 70)
        self.dp1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(70, 50)
        self.dp2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(50, 20)
        self.dp3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(20, 9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dp1(x)

        x = self.fc2(x)
        x = self.dp2(x)

        x = self.fc3(x)
        x = self.dp3(x)

        x = self.fc4(x)
        return x


class Dataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        data = self.X[idx]
        target = self.y[idx][0]
        return data, target

    def __len__(self):
        return len(self.X)


def load_dataset(file_name, columns, activities):
    df = pd.read_csv(file_name, header=None)
    df.columns = ["yaw", "pitch", "acc_x", "acc_y", "acc_z", "activityName"]
    df["activity"] = df["activityName"].replace(activities, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    return df


def convert_to_timeseries(features, num_time_steps, num_features):
    data = np.reshape(features, newshape=(num_time_steps, num_features))
    return data.T


def train_test_split_dataset(df, random_state=42):
    X, y = list(), list()
    df_len = len(df)
    for idx in range(0, df_len, 30):
        window_df = df[idx : idx + 60]
        labels = window_df["activity"].unique()
        if len(labels) != 1 or len(window_df) < 60:
            continue
        assert len(labels) == 1 and len(window_df) == 60
        features = window_df.drop(columns=["activity", "activityName"]).values
        features = convert_to_timeseries(features, num_time_steps=60, num_features=5)
        X.append(features)
        y.append(labels)

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=random_state
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def prepare_dataloader(X_train, X_valid, X_test, y_train, y_valid, y_test):
    traindataset = Dataset(X_train, y_train)

    trainloader = torch.utils.data.DataLoader(
        traindataset, batch_size=100, shuffle=True, num_workers=4,
    )

    validdataset = Dataset(X_valid, y_valid)

    validloader = torch.utils.data.DataLoader(
        validdataset, batch_size=100, shuffle=True, num_workers=4,
    )

    testdataset = Dataset(X_test, y_test)

    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=100, shuffle=True, num_workers=4,
    )

    return trainloader, validloader, testloader


def load_dataloader():
    activities = [
        "elbow_lock",
        "hair",
        "pushback",
        "rocket",
        "scarecrow",
        "shouldershrug",
        "windows",
        "zigzag",
        "logout",
    ]
    columns = ["yaw", "pitch", "acc_x", "acc_y", "acc_z", "activityName"]
    file_name = "prelim_dataset.csv"

    df = load_dataset(file_name, columns, activities)
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_split_dataset(df)
    trainloader, validloader, testloader = prepare_dataloader(
        X_train, X_valid, X_test, y_train, y_valid, y_test
    )
    return trainloader, validloader, testloader


if __name__ == "__main__":
    trainloader, validloader, testloader = load_dataloader()
    inputs, labels = next(iter(trainloader))
    print(inputs[:1], labels[:1])
    print(inputs[:1].shape)
