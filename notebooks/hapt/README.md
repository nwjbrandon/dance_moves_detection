# Human Activity Recognition

This notebook evalutes different machine learning models to predict the 6 human activities walking, walking upstairs, walking downstairs, sitting, standing or laying from accelerometer and gyroscope readings. Refer to the jupyter notebook for more details.

# Models
Different models are explored to classify the human activities. It is trained on a time series data from [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) (HARUS) from UCI.
- SVC: Support vector classifier trained on full 561 features non-time series data
- DNN: Neural network consisting of linear layers trained on full 561 features non-time series data
- MSVC: Support vector classifier trained on selected 36 features non-time series data
- MDNN: Neural network consisting of linear layers trained on 36 features non-time series data
- CNN: Neural network consisting convolutional 1d layer trained on time series data
- TDNN: Neural network without convolutional 1d layer trained on time series data

# Performance
| model                        | acc_val        | acc_test       |      
| ---------------------------- | :------------: | :------------: |
| SVC                          | 0.986042       | 0.987379       |
| DNN                          | 0.968932       | 0.972330       |
| MSVC                         | 0.898532       | 0.906796       |
| MDNN                         | 0.864563       | 0.861165       |
| CNN                          | 0.959223       | 0.952427       |
| TDNN                         | 0.923301       | 0.923301       |

# Conclusion
SVC and DNN models performed relatively well achieve comparatively high validation and testing accuracy. CNN achieved accuracy less than 2% compared to SVC and DNN. This is likely attributed to the more complex features that the CNN has to learn from the raw data. However, CNN model is simple as the model can learn on the training data without the need for tedious, domain-expertised feature extractions.

# Reference
- https://github.com/UdiBhaskar/Human-Activity-Recognition--Using-Deep-NN
- https://github.com/bobbleoxs/data_science
- https://github.com/jchiang2/Human-Activity-Recognition