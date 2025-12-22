#pragma once

struct Config {
    // Model parameters
    int regression_hidden_size = 512;
    int classification_hidden_size = 256;
    float learning_rate = 0.001;
    int batch_size = 16;
    int epochs = 300;
    // Separate epochs for regression to allow longer training when needed
    int regression_epochs = 600;
    
    // Feature configuration
    struct Features {
        bool use_numerical = true;
        bool use_categorical = true;
        bool use_temporal = true;
        bool use_derived = true;
    } features;
    
    // Data paths
    std::string data_path = "Inventory Management E-Grocery - InventoryData.csv";
    std::string model_save_path = "inventory_models/";
    
    // Validation split
    float train_split = 0.8;  // 800 samples for training
    float val_split = 0.1;    // 100 samples for validation
    float test_split = 0.1;   // 100 samples for testing
};
