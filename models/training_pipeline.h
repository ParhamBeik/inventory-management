#pragma once
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include "config.h"
#include "data_preprocessor.h"
#include "models.h"

class TrainingPipeline {
private:
    Config config;
    std::shared_ptr<DataPreprocessor> preprocessor;
    
public:
    TrainingPipeline(const Config& cfg) : config(cfg) {
        preprocessor = std::make_shared<DataPreprocessor>();
    }
    
    void run() {
        std::cout << "=== Inventory Management ML Pipeline ===\n";
        std::cout << "Loading and preprocessing data...\n";
        
        auto records = preprocessor->loadAndPreprocess(config.data_path, 1000);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(records.begin(), records.end(), g);
        
        int total_samples = records.size();
        int train_samples = total_samples * config.train_split;
        int val_samples = total_samples * config.val_split;
        
        std::vector<InventoryRecord> train_records(records.begin(), records.begin() + train_samples);
        std::vector<InventoryRecord> val_records(records.begin() + train_samples, 
                                                records.begin() + train_samples + val_samples);
        std::vector<InventoryRecord> test_records(records.begin() + train_samples + val_samples, 
                                                 records.end());
        
        std::cout << "Data split: " << train_samples << " train, " 
                  << val_samples << " val, " << test_records.size() << " test\n";
        
        auto [train_features_reg, train_targets_reg] = preprocessor->prepareRegressionData(train_records);
        auto [val_features_reg, val_targets_reg] = preprocessor->prepareRegressionData(val_records);
        auto [test_features_reg, test_targets_reg] = preprocessor->prepareRegressionData(test_records);
        
        auto [train_features_cls, train_targets_cls] = preprocessor->prepareClassificationData(train_records);
        auto [val_features_cls, val_targets_cls] = preprocessor->prepareClassificationData(val_records);
        auto [test_features_cls, test_targets_cls] = preprocessor->prepareClassificationData(test_records);
        
        preprocessor->normalizeFeatures(train_features_reg);
        preprocessor->normalizeFeatures(val_features_reg);
        preprocessor->normalizeFeatures(test_features_reg);
        
        preprocessor->normalizeFeatures(train_features_cls);
        preprocessor->normalizeFeatures(val_features_cls);
        preprocessor->normalizeFeatures(test_features_cls);
        
        std::cout << "\n=== Training Regression Model (Demand Forecasting) ===\n";
        trainRegressionModel(train_features_reg, train_targets_reg, 
                           val_features_reg, val_targets_reg);
        
        std::cout << "\n=== Training Classification Model (Inventory Status) ===\n";
        trainClassificationModel(train_features_cls, train_targets_cls,
                               val_features_cls, val_targets_cls);
        
        std::cout << "\n=== Final Evaluation on Test Set ===\n";
        evaluateModels(test_features_reg, test_targets_reg,
                      test_features_cls, test_targets_cls);
        
        // Save encodings for inference
        preprocessor->saveEncodings(config.model_save_path + "encodings.txt");
    }
    
private:
    void trainRegressionModel(const torch::Tensor& train_features, const torch::Tensor& train_targets,
                            const torch::Tensor& val_features, const torch::Tensor& val_targets) {
        int input_size = train_features.size(1);
        RegressionModel model(input_size, config.regression_hidden_size);
        
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config.learning_rate).weight_decay(1e-5));
        auto scheduler = torch::optim::StepLR(optimizer, 40, 0.7);
        
        int best_epoch = 0;
        float best_val_loss = std::numeric_limits<float>::max();
        
        for (int epoch = 0; epoch < config.regression_epochs; ++epoch) {
            model->train();
            
            int num_batches = ceil(train_features.size(0) / float(config.batch_size));
            float total_loss = 0.0f;
            
            for (int batch = 0; batch < num_batches; ++batch) {
                int start_idx = batch * config.batch_size;
                int end_idx = std::min(start_idx + config.batch_size, (int)train_features.size(0));
                
                auto batch_features = train_features.index({torch::indexing::Slice(start_idx, end_idx)});
                auto batch_targets = train_targets.index({torch::indexing::Slice(start_idx, end_idx)});
                
                optimizer.zero_grad();
                auto predictions = model->forward(batch_features).squeeze();
                auto loss = torch::l1_loss(predictions, batch_targets);  // MAE instead of MSE
                
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                optimizer.step();
                
                total_loss += loss.item<float>();
            }
            
            model->eval();
            auto val_predictions = model->forward(val_features).squeeze();
            auto val_loss = torch::l1_loss(val_predictions, val_targets).item<float>();
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                best_epoch = epoch;
                torch::save(model, config.model_save_path + "regression_model_es.pt");
            }
            
            scheduler.step();
            
            if (epoch % 15 == 0) {
                std::cout << "Epoch " << epoch << " | Train MAE: " << (total_loss / num_batches)
                          << " | Val MAE: " << val_loss << std::endl;
            }
        }
        torch::save(model, config.model_save_path + "regression_model.pt");
        std::cout << "Best model at epoch " << best_epoch << " with Val MAE: " << best_val_loss << std::endl;
    }
    
    void trainClassificationModel(const torch::Tensor& train_features, const torch::Tensor& train_targets,
                                const torch::Tensor& val_features, const torch::Tensor& val_targets) {
        int input_size = train_features.size(1);
        ClassificationModel model(input_size, 3, config.classification_hidden_size);
        
        torch::optim::Adam optimizer(model->parameters(), config.learning_rate);
        auto scheduler = torch::optim::StepLR(optimizer, 30, 0.5);
        
        int best_epoch = 0;
        float best_val_acc = 0.0f;
        
        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            model->train();
            
            int num_batches = ceil(train_features.size(0) / float(config.batch_size));
            float total_loss = 0.0f;
            int correct = 0;
            int total = 0;
            
            for (int batch = 0; batch < num_batches; ++batch) {
                int start_idx = batch * config.batch_size;
                int end_idx = std::min(start_idx + config.batch_size, (int)train_features.size(0));
                
                auto batch_features = train_features.index({torch::indexing::Slice(start_idx, end_idx)});
                auto batch_targets = train_targets.index({torch::indexing::Slice(start_idx, end_idx)});
                
                optimizer.zero_grad();
                auto predictions = model->forward(batch_features);
                auto loss = torch::nll_loss(predictions, batch_targets);
                
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                optimizer.step();
                
                total_loss += loss.item<float>();
                
                auto pred_classes = torch::argmax(predictions, 1);
                correct += torch::sum(pred_classes == batch_targets).item<int>();
                total += batch_targets.size(0);
            }
            
            model->eval();
            auto val_predictions = model->forward(val_features);
            auto val_loss = torch::nll_loss(val_predictions, val_targets).item<float>();
            auto val_pred_classes = torch::argmax(val_predictions, 1);
            float val_acc = torch::sum(val_pred_classes == val_targets).item<float>() / val_targets.size(0);
            
            if (val_acc > best_val_acc) {
                best_val_acc = val_acc;
                best_epoch = epoch;
                torch::save(model, config.model_save_path + "classification_model.pt");
            }
            
            scheduler.step();
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << " | Train Loss: " << (total_loss / num_batches)
                          << " | Train Acc: " << (100.0f * correct / total) << "%"
                          << " | Val Loss: " << val_loss
                          << " | Val Acc: " << (100.0f * val_acc) << "%" << std::endl;
            }
        }
        
        std::cout << "Best model at epoch " << best_epoch << " with Val Acc: " << (100.0f * best_val_acc) << "%" << std::endl;
    }
    
    void evaluateModels(const torch::Tensor& test_features_reg, const torch::Tensor& test_targets_reg,
                       const torch::Tensor& test_features_cls, const torch::Tensor& test_targets_cls) {
        
        RegressionModel reg_model(test_features_reg.size(1));
        ClassificationModel cls_model(test_features_cls.size(1));

        
        torch::load(reg_model, config.model_save_path + "regression_model.pt");
        torch::load(cls_model, config.model_save_path + "classification_model.pt");
        
        
        reg_model->eval();


        auto reg_predictions = reg_model->forward(test_features_reg).squeeze();
        // Convert from log space back to original scale for metrics
        auto reg_predictions_original = reg_predictions;
        auto reg_targets_original = test_targets_reg;
        
        auto diff = reg_predictions_original - reg_targets_original;
        auto reg_mae = torch::mean(torch::abs(diff)).item<float>();
        auto reg_rmse = torch::sqrt(torch::mean(diff * diff)).item<float>();
        // Mean Absolute Percentage Error (MAPE) - avoid division by zero
        float eps = 1e-6f;
        auto pct_errors = torch::abs(diff / (reg_targets_original + eps));
        auto reg_mape = torch::mean(pct_errors).item<float>() * 100.0f;
        auto target_mean = reg_targets_original.mean();
        auto ss_total = torch::sum(torch::pow(reg_targets_original - target_mean, 2)).item<float>();
        auto ss_residual = torch::sum(torch::pow(diff, 2)).item<float>();
        float r2 = 1.0f - (ss_residual / (ss_total + 1e-6));
        
        std::cout << "\nRegression Model Results:\n";
        std::cout << "RMSE: " << reg_rmse << "\n";
        std::cout << "MAE: " << reg_mae << "\n";
        std::cout << "MAPE (%): " << reg_mape << "%\n";
        std::cout << "R² Score: " << r2 << "\n";
        
        cls_model->eval();
        auto cls_predictions = cls_model->forward(test_features_cls);
        auto cls_pred_classes = torch::argmax(cls_predictions, 1);
        
        float accuracy = torch::sum(cls_pred_classes == test_targets_cls).item<float>() / test_targets_cls.size(0);
        
        std::vector<int> tp(3, 0), fp(3, 0), fn(3, 0);
        
        for (int i = 0; i < test_targets_cls.size(0); ++i) {
            int pred = cls_pred_classes[i].item<int>();
            int actual = test_targets_cls[i].item<int>();
            
            if (pred == actual) {
                tp[pred]++;
            } else {
                fp[pred]++;
                fn[actual]++;
            }
        }
        
        std::cout << "\nClassification Model Results:\n";
        std::cout << "Overall Accuracy: " << (100.0f * accuracy) << "%\n";
        
        std::vector<std::string> class_names = {"In Stock", "Low Stock", "Expiring Soon"};
        for (int i = 0; i < 3; ++i) {
            float precision = (tp[i] + fp[i] > 0) ? tp[i] / float(tp[i] + fp[i]) : 0.0f;
            float recall = (tp[i] + fn[i] > 0) ? tp[i] / float(tp[i] + fn[i]) : 0.0f;
            float f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0f;
            
            std::cout << class_names[i] << " - Precision: " << precision
                      << ", Recall: " << recall << ", F1: " << f1 << "\n";
        }
        
        savePredictions(reg_predictions, test_targets_reg, cls_pred_classes, test_targets_cls);
    }
    
    void savePredictions(const torch::Tensor& reg_pred, const torch::Tensor& reg_true,
                        const torch::Tensor& cls_pred, const torch::Tensor& cls_true) {
        
        std::ofstream file(config.model_save_path + "predictions.csv");
        file << "Reg_Prediction,Reg_Actual,Cls_Prediction,Cls_Actual\n";
        
        for (int i = 0; i < reg_pred.size(0); ++i) {
            
            file << reg_pred[i].item<float>() << ","
                 << reg_true[i].item<float>() << ","
                 << cls_pred[i].item<int>() << ","
                 << cls_true[i].item<int>() << "\n";
        }
        
        file.close();
        std::cout << "\nPredictions saved to: " << config.model_save_path + "predictions.csv\n";
    }
};
