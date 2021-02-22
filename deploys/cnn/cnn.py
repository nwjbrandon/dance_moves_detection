import warnings

import numpy as np
import torch
from joblib import load
from utils import CNN, load_cnn_dataloader

warnings.filterwarnings("ignore")


def scale_data(data, scaler, is_train=False):
    """
        data: inputs of shape (num_instances, num_features, num_time_steps)
        scaler: standard scalar to scale data
    """
    num_instances, num_time_steps, num_features = data.shape
    data = np.reshape(data, newshape=(-1, num_features))
    if is_train:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    data = np.reshape(data, newshape=(num_instances, num_time_steps, num_features))
    return data


def main():
    # Prepare model
    model = CNN()
    model.load_state_dict(torch.load("prelim_cnn.pth"))
    model.eval()

    # Scale data
    scaler = load("std_scaler.bin")

    # Prepare data
    _, _, dataloader = load_cnn_dataloader()

    # Run inference
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = scale_data(inputs, scaler)
        inputs = torch.tensor(inputs)
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy: ", correct / total)


if __name__ == "__main__":
    main()
