#pragma once
#include <torch/torch.h>

struct RegressionModelImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Dropout dropout{nullptr};
    
    RegressionModelImpl() = default;
    
    RegressionModelImpl(int input_size, int hidden_size = 512) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        bn1 = register_module("bn1", torch::nn::BatchNorm1d(hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size/2));
        bn2 = register_module("bn2", torch::nn::BatchNorm1d(hidden_size/2));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size/2, hidden_size/4));
        bn3 = register_module("bn3", torch::nn::BatchNorm1d(hidden_size/4));
        fc4 = register_module("fc4", torch::nn::Linear(hidden_size/4, 1));
        dropout = register_module("dropout", torch::nn::Dropout(0.2));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1(fc1(x)));
        x = dropout(x);
        x = torch::relu(bn2(fc2(x)));
        x = dropout(x);
        x = torch::relu(bn3(fc3(x)));
        x = dropout(x);
        x = fc4(x);
        return x;
    }
};

TORCH_MODULE(RegressionModel);

struct ClassificationModelImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr};
    torch::nn::Dropout dropout{nullptr};
    
    ClassificationModelImpl() = default;
    
    ClassificationModelImpl(int input_size, int num_classes = 3, int hidden_size = 256) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        bn1 = register_module("bn1", torch::nn::BatchNorm1d(hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size/2));
        bn2 = register_module("bn2", torch::nn::BatchNorm1d(hidden_size/2));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size/2, num_classes));
        dropout = register_module("dropout", torch::nn::Dropout(0.4));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1(fc1(x)));
        x = dropout(x);
        x = torch::relu(bn2(fc2(x)));
        x = dropout(x);
        x = fc3(x);
        return torch::log_softmax(x, 1);
    }
};

TORCH_MODULE(ClassificationModel);
