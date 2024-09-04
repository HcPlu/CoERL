# -*- coding:utf-8 _*-
"""
=================================================
@Project -> File ：CCERL -> util.py
@Author ：HcPlu
@Version: 1.0
@Date: 2023/9/8 13:50
@@Description: 
==================================================
"""

import torch
import torch.nn as nn
import numpy as np

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Create an instance of the neural network


# Flatten and encode the model's parameters
def encode_model(model):
    flattened_params = np.concatenate([param.cpu().detach().numpy().flatten() for param in model.parameters()])
    return flattened_params



# Decode the vector and rebuild the model
def decode_model( model,noise,noise_std):
    decoded_model = model
    # add noise
    start = 0
    individual_noise = []
    for param in decoded_model.parameters():
        num_params = np.prod(param.shape)
        reshaped_noise = noise[start:start+num_params].reshape(param.shape)
        param.data = param.data+reshaped_noise*noise_std
        individual_noise.append(reshaped_noise)
        start += num_params
    return decoded_model,individual_noise

def direct_decode_model( model,noise):
    decoded_model = model

    start = 0
    individual_noise = []
    for param in decoded_model.parameters():
        num_params = np.prod(param.shape)
        reshaped_noise = noise[start:start+num_params].reshape(param.shape)
        param.data = reshaped_noise
        individual_noise.append(reshaped_noise)
        start += num_params
    return decoded_model,individual_noise

def hard_copy(a,b):
    #update model a's parameters with model b's parameters
    for param_a,param_b in zip(a.parameters(),b.parameters()):
        param_a.data = param_b.data.clone()


# def decode_model_shape(encoded_vector, shapes):
#     start = 0
#     for param in shapes:
#         num_params = np.prod(param)
#         param.data = torch.tensor(encoded_vector[start:start+num_params].reshape(param))
#         start += num_params
#     return decoded_model



# if __name__== "__main__":
#     model = SimpleNet()
#     encoded_vector = encode_model(model)
#     decoded_model = decode_model(encoded_vector, model)
#
#     # Verify that the decoded model's parameters match the original model's parameters
#     for orig_param, decoded_param in zip(model.parameters(), decoded_model.parameters()):
#         print(orig_param.shape)
#         assert torch.allclose(orig_param, decoded_param, atol=1e-5)