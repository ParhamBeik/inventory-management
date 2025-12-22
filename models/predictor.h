#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include "data_preprocessor.h"
#include "models.h"

class InventoryPredictor {
private:
    std::shared_ptr<DataPreprocessor> preprocessor;
    RegressionModel reg_model;
    ClassificationModel cls_model;
    bool models_loaded = false;
    int num_features = 0;
    
public:
    InventoryPredictor() {
        preprocessor = std::make_shared<DataPreprocessor>();
    }
    
    bool loadModels(const std::string& model_path, int input_size) {
        try {
            reg_model = RegressionModel(input_size);
            cls_model = ClassificationModel(input_size);
            
            torch::load(reg_model, model_path + "regression_model.pt");
            torch::load(cls_model, model_path + "classification_model.pt");
            
            // Load encodings
            preprocessor->loadEncodings(model_path + "encodings.txt");
            
            models_loaded = true;
            num_features = input_size;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading models: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::pair<float, std::string> predictSingle(const InventoryRecord& record) {
        if (!models_loaded) {
            throw std::runtime_error("Models not loaded");
        }
        
        std::vector<InventoryRecord> single_record = {record};
        auto [features_reg, _] = preprocessor->prepareRegressionData(single_record);
        auto [features_cls, __] = preprocessor->prepareClassificationData(single_record);
        
        preprocessor->normalizeFeatures(features_reg);
        preprocessor->normalizeFeatures(features_cls);
        
        reg_model->eval();
        cls_model->eval();
        
        auto reg_pred = reg_model->forward(features_reg).squeeze().item<float>();
            // Convert from log space back to original scale
            //reg_pred = exp(reg_pred) - 1.0f;
        auto cls_logits = cls_model->forward(features_cls);
        auto cls_pred = torch::argmax(cls_logits, 1).item<int>();
        
        std::vector<std::string> status_names = {"In Stock", "Low Stock", "Expiring Soon"};
        std::string status_name = (cls_pred >= 0 && cls_pred < 3) ? status_names[cls_pred] : "Unknown";
        
        return {reg_pred, status_name};
    }
    
    std::vector<std::pair<float, std::string>> predictBatch(const std::vector<InventoryRecord>& records) {
        std::vector<std::pair<float, std::string>> predictions;
        
        auto [features_reg, _] = preprocessor->prepareRegressionData(records);
        auto [features_cls, __] = preprocessor->prepareClassificationData(records);
        
        preprocessor->normalizeFeatures(features_reg);
        preprocessor->normalizeFeatures(features_cls);
        
        reg_model->eval();
        cls_model->eval();
        
        auto reg_preds = reg_model->forward(features_reg).squeeze();
        auto cls_logits = cls_model->forward(features_cls);
        auto cls_preds = torch::argmax(cls_logits, 1);
        
        std::vector<std::string> status_names = {"In Stock", "Low Stock", "Expiring Soon"};
        
        for (int i = 0; i < records.size(); ++i) {
            int cls_idx = cls_preds[i].item<int>();
            std::string status = (cls_idx >= 0 && cls_idx < 3) ? status_names[cls_idx] : "Unknown";
                // Convert from log space back to original scale
                //float original_pred = exp(reg_preds[i].item<float>()) - 1.0f;
                predictions.emplace_back(reg_preds[i].item<float>(), status);
        }
        
        return predictions;
    }
};
