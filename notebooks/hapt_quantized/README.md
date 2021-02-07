# Human Activity Recognition

This notebook evalutes different quantized methods on CNN mdoel to predict the 6 human activities walking, walking upstairs, walking downstairs, sitting, standing or laying from accelerometer and gyroscope readings. Refer to the jupyter notebook for more details.

# Quantized Models
A CNN model is selected to classify the human activities. It is trained on a time series data from [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) (HARUS) from UCI. There are three methods to quantized the models.
- Dynamic Quantization
- Static Quantization
- Quantization Aware Trained

# Performance
| model                        | acc_test       | latency(ms)    |      
| ---------------------------- | :------------: | :------------: |
| Original                     | 0.964          | 5.85           |
| Dynamic Quantization         | 0.967          | 5.21           |
| Static Quantization          | 0.950          | 3.06           |
| Quantization Aware Trained   | 0.946          | 2.92           |


# Conclusion
All three quantizated model acheive comparatively high accuracies that is less by at most 1%. The quantized awared trained model achieve the shortest inference latency. The dynamic quantization method could only quantized Linear and LSTM layers only, and therefore, do not have much difference in latency and accuracy as compared to the original.

# Reference
- https://github.com/leimao/PyTorch-Quantization-Aware-Training
- https://github.com/leimao/PyTorch-Static-Quantization
- https://github.com/leimao/PyTorch-Dynamic-Quantization