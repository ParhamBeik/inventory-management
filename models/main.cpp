#include <torch/torch.h>
#include <iostream>
#include <algorithm>
#include <random>

#include "config.h"
#include "data_structures.h"
#include "data_preprocessor.h"
#include "models.h"
#include "training_pipeline.h"
#include "predictor.h"

// ==================== Main Function ====================
int main() {
    try {
        Config config;
        config.data_path = "Inventory Management E-Grocery - InventoryData.csv";
        
        system("mkdir -p inventory_models");
        
        auto pipeline = std::make_unique<TrainingPipeline>(config);
        pipeline->run();
        
        std::cout << "\n=== Example Inference ===" << std::endl;
        
        InventoryPredictor predictor;
        
        DataPreprocessor dp;
        auto sample_records = dp.loadAndPreprocess(config.data_path, 1000);
        if (!sample_records.empty()) {
            int num_features = dp.getNumFeatures();
            if (predictor.loadModels(config.model_save_path, num_features)) {
                auto [demand_pred, status_pred] = predictor.predictSingle(sample_records[0]);
                std::cout << "Predicted Demand (next 30 days): " << demand_pred << std::endl;
                std::cout << "Predicted Inventory Status: " << status_pred << std::endl;
                std::cout << "\n=== Batch Prediction (first 5 samples) ===" << std::endl;
                std::vector<InventoryRecord> batch_samples(sample_records.begin(), sample_records.begin() + std::min(5, (int)sample_records.size()));
                auto batch_predictions = predictor.predictBatch(batch_samples);
                
                for (size_t i = 0; i < batch_predictions.size(); ++i) {
                    std::cout << "Sample " << i + 1 << ": Demand = " << batch_predictions[i].first 
                              << ", Status = " << batch_predictions[i].second << std::endl;
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}