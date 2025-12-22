#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <torch/torch.h>
#include "data_structures.h"

class DataPreprocessor {
public:
    std::map<std::string, int> category_encoding;
    std::map<std::string, int> abc_encoding;
    std::map<std::string, int> status_encoding;
    std::map<std::string, int> warehouse_encoding;
    
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    
private:
    
    int parseDateToDays(const std::string& date_str) {
        if (date_str.empty()) return 0;
        
        int year, month, day;
        char dash1, dash2;
        std::stringstream ss(date_str);
        ss >> year >> dash1 >> month >> dash2 >> day;
        
        return (year - 2020) * 365 + (month - 1) * 30 + day;
    }
    
    float parseEuropeanNumber(const std::string& num_str) {
        if (num_str.empty()) return 0.0f;
        
        std::string cleaned;
        for (char c : num_str) {
            if (c == ',') cleaned += '.';
            else if (c == '.' || c == '$' || c == '%') continue;
            else if (c >= '0' && c <= '9') cleaned += c;
        }
        
        try {
            return std::stof(cleaned);
        } catch (...) {
            return 0.0f;
        }
    }

public:
    DataPreprocessor() = default;
    
    std::vector<InventoryRecord> loadAndPreprocess(const std::string& filename, int max_samples = 1000) {
        std::vector<InventoryRecord> records;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open data file: " + filename);
        }
        
        std::string line;
        std::getline(file, line);
        
        int sample_count = 0;
        
        while (std::getline(file, line) && sample_count < max_samples) {
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            
            bool in_quotes = false;
            std::string field;
            
            for (char c : line) {
                if (c == '"') {
                    in_quotes = !in_quotes;
                } else if (c == ',' && !in_quotes) {
                    tokens.push_back(field);
                    field.clear();
                } else {
                    field += c;
                }
            }
            tokens.push_back(field);
            
            if (tokens.size() < 37) continue;
            
            InventoryRecord record;
            
            record.sku_id = tokens[0];
            record.sku_name = tokens[1];
            record.category = tokens[2];
            record.abc_class = tokens[3];
            record.supplier_id = tokens[4];
            record.warehouse_id = tokens[6];
            record.inventory_status = tokens[31];
            record.fifo_fefo = tokens[30];
            
            record.stock_age_days = parseEuropeanNumber(tokens[12]);
            record.quantity_on_hand = std::stoi(tokens[13]);
            record.quantity_reserved = std::stoi(tokens[14]);
            record.quantity_committed = std::stoi(tokens[15]);
            record.damaged_qty = std::stoi(tokens[16]);
            record.returns_qty = std::stoi(tokens[17]);
            record.avg_daily_sales = parseEuropeanNumber(tokens[18]);
            record.forecast_next_30d = parseEuropeanNumber(tokens[19]);
            record.days_of_inventory = parseEuropeanNumber(tokens[20]);
            record.reorder_point = std::stoi(tokens[21]);
            record.safety_stock = std::stoi(tokens[22]);
            record.lead_time_days = std::stoi(tokens[23]);
            record.unit_cost_usd = parseEuropeanNumber(tokens[24]);
            record.last_purchase_price_usd = parseEuropeanNumber(tokens[25]);
            record.total_inventory_value_usd = parseEuropeanNumber(tokens[26]);
            record.sku_churn_rate = parseEuropeanNumber(tokens[27]);
            record.order_frequency_per_month = parseEuropeanNumber(tokens[28]);
            record.supplier_ontime_pct = parseEuropeanNumber(tokens[29]);
            record.count_variance = std::stoi(tokens[32]);
            record.audit_variance_pct = parseEuropeanNumber(tokens[34]);
            record.demand_forecast_accuracy_pct = parseEuropeanNumber(tokens[35]);
            
            record.received_date_days = parseDateToDays(tokens[9]);
            record.last_purchase_date_days = parseDateToDays(tokens[10]);
            record.expiry_date_days = parseDateToDays(tokens[11]);
            record.audit_date_days = parseDateToDays(tokens[33]);
            
            record.regression_target = record.forecast_next_30d;
            
            if (record.inventory_status == "In Stock") record.classification_target = 0;
            else if (record.inventory_status == "Low Stock") record.classification_target = 1;
            else if (record.inventory_status == "Expiring Soon") record.classification_target = 2;
            else record.classification_target = 0;
            
            records.push_back(record);
            sample_count++;
        }
        
        buildEncodings(records);
        
        return records;
    }
    
    void buildEncodings(const std::vector<InventoryRecord>& records) {
        int cat_idx = 0, abc_idx = 0, status_idx = 0, wh_idx = 0;
        
        for (const auto& record : records) {
            if (category_encoding.find(record.category) == category_encoding.end()) {
                category_encoding[record.category] = cat_idx++;
            }
            if (abc_encoding.find(record.abc_class) == abc_encoding.end()) {
                abc_encoding[record.abc_class] = abc_idx++;
            }
            if (status_encoding.find(record.inventory_status) == status_encoding.end()) {
                status_encoding[record.inventory_status] = status_idx++;
            }
            if (warehouse_encoding.find(record.warehouse_id) == warehouse_encoding.end()) {
                warehouse_encoding[record.warehouse_id] = wh_idx++;
            }
        }
    }
    
    int getNumFeatures() const {
        return 20 + 6 + category_encoding.size() + abc_encoding.size() + warehouse_encoding.size() + 1;
    }
    
    std::pair<torch::Tensor, torch::Tensor> prepareRegressionData(const std::vector<InventoryRecord>& records) {
        int num_samples = records.size();
        int num_features = getNumFeatures();
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor features = torch::zeros({num_samples, num_features}, options);
        torch::Tensor targets = torch::zeros({num_samples}, options);
        
        for (int i = 0; i < num_samples; ++i) {
            const auto& record = records[i];
            int feature_idx = 0;
            
            features[i][feature_idx++] = record.stock_age_days;
            features[i][feature_idx++] = record.quantity_on_hand;
            features[i][feature_idx++] = record.quantity_reserved;
            features[i][feature_idx++] = record.quantity_committed;
            features[i][feature_idx++] = record.damaged_qty;
            features[i][feature_idx++] = record.returns_qty;
            features[i][feature_idx++] = record.avg_daily_sales;
            features[i][feature_idx++] = record.days_of_inventory;
            features[i][feature_idx++] = record.reorder_point;
            features[i][feature_idx++] = record.safety_stock;
            features[i][feature_idx++] = record.lead_time_days;
            features[i][feature_idx++] = record.unit_cost_usd;
            features[i][feature_idx++] = record.last_purchase_price_usd;
            features[i][feature_idx++] = record.total_inventory_value_usd;
            features[i][feature_idx++] = record.sku_churn_rate;
            features[i][feature_idx++] = record.order_frequency_per_month;
            features[i][feature_idx++] = record.supplier_ontime_pct;
            features[i][feature_idx++] = record.count_variance;
            features[i][feature_idx++] = record.audit_variance_pct;
            features[i][feature_idx++] = record.demand_forecast_accuracy_pct;
            
            features[i][feature_idx++] = record.quantity_on_hand / (record.avg_daily_sales + 1e-6);
            features[i][feature_idx++] = record.stock_age_days / (record.lead_time_days + 1e-6);
            features[i][feature_idx++] = (record.expiry_date_days - record.received_date_days);
            features[i][feature_idx++] = (record.expiry_date_days - record.last_purchase_date_days);
            features[i][feature_idx++] = record.quantity_reserved / (record.quantity_on_hand + 1e-6);
            features[i][feature_idx++] = (record.damaged_qty + record.returns_qty);
            
            int cat_encoded = (category_encoding.count(record.category) > 0) ? category_encoding[record.category] : 0;
            for (int j = 0; j < category_encoding.size(); ++j) {
                features[i][feature_idx++] = (j == cat_encoded) ? 1.0f : 0.0f;
            }
            
            int abc_encoded = (abc_encoding.count(record.abc_class) > 0) ? abc_encoding[record.abc_class] : 0;
            for (int j = 0; j < abc_encoding.size(); ++j) {
                features[i][feature_idx++] = (j == abc_encoded) ? 1.0f : 0.0f;
            }
            
            int wh_encoded = (warehouse_encoding.count(record.warehouse_id) > 0) ? warehouse_encoding[record.warehouse_id] : 0;
            for (int j = 0; j < warehouse_encoding.size(); ++j) {
                features[i][feature_idx++] = (j == wh_encoded) ? 1.0f : 0.0f;
            }
            
            features[i][feature_idx++] = (record.fifo_fefo == "FIFO") ? 1.0f : 0.0f;
            
            targets[i] = record.regression_target;
        }
        
        return {features, targets};
    }
    
    std::pair<torch::Tensor, torch::Tensor> prepareClassificationData(const std::vector<InventoryRecord>& records) {
        int num_samples = records.size();
        int num_features = getNumFeatures();
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor features = torch::zeros({num_samples, num_features}, options);
        torch::Tensor targets = torch::zeros({num_samples}, torch::kLong);
        
        for (int i = 0; i < num_samples; ++i) {
            const auto& record = records[i];
            int feature_idx = 0;
            
            features[i][feature_idx++] = record.stock_age_days;
            features[i][feature_idx++] = record.quantity_on_hand;
            features[i][feature_idx++] = record.quantity_reserved;
            features[i][feature_idx++] = record.quantity_committed;
            features[i][feature_idx++] = record.damaged_qty;
            features[i][feature_idx++] = record.returns_qty;
            features[i][feature_idx++] = record.avg_daily_sales;
            features[i][feature_idx++] = record.days_of_inventory;
            features[i][feature_idx++] = record.reorder_point;
            features[i][feature_idx++] = record.safety_stock;
            features[i][feature_idx++] = record.lead_time_days;
            features[i][feature_idx++] = record.unit_cost_usd;
            features[i][feature_idx++] = record.last_purchase_price_usd;
            features[i][feature_idx++] = record.total_inventory_value_usd;
            features[i][feature_idx++] = record.sku_churn_rate;
            features[i][feature_idx++] = record.order_frequency_per_month;
            features[i][feature_idx++] = record.supplier_ontime_pct;
            features[i][feature_idx++] = record.count_variance;
            features[i][feature_idx++] = record.audit_variance_pct;
            features[i][feature_idx++] = record.demand_forecast_accuracy_pct;
            
            features[i][feature_idx++] = record.quantity_on_hand / (record.avg_daily_sales + 1e-6);
            features[i][feature_idx++] = record.stock_age_days / (record.lead_time_days + 1e-6);
            features[i][feature_idx++] = (record.expiry_date_days - record.received_date_days);
            features[i][feature_idx++] = (record.expiry_date_days - record.last_purchase_date_days);
            features[i][feature_idx++] = record.quantity_reserved / (record.quantity_on_hand + 1e-6);
            features[i][feature_idx++] = (record.damaged_qty + record.returns_qty);
            
            int cat_encoded = (category_encoding.count(record.category) > 0) ? category_encoding[record.category] : 0;
            for (int j = 0; j < category_encoding.size(); ++j) {
                features[i][feature_idx++] = (j == cat_encoded) ? 1.0f : 0.0f;
            }
            
            int abc_encoded = (abc_encoding.count(record.abc_class) > 0) ? abc_encoding[record.abc_class] : 0;
            for (int j = 0; j < abc_encoding.size(); ++j) {
                features[i][feature_idx++] = (j == abc_encoded) ? 1.0f : 0.0f;
            }
            
            int wh_encoded = (warehouse_encoding.count(record.warehouse_id) > 0) ? warehouse_encoding[record.warehouse_id] : 0;
            for (int j = 0; j < warehouse_encoding.size(); ++j) {
                features[i][feature_idx++] = (j == wh_encoded) ? 1.0f : 0.0f;
            }
            
            features[i][feature_idx++] = (record.fifo_fefo == "FIFO") ? 1.0f : 0.0f;
            
            targets[i] = record.classification_target;
        }
        
        return {features, targets};
    }
    
    void normalizeFeatures(torch::Tensor& features) {
        if (feature_means.empty()) {
            feature_means = std::vector<float>(features.size(1), 0.0f);
            feature_stds = std::vector<float>(features.size(1), 1.0f);
            
            for (int i = 0; i < features.size(1); ++i) {
                auto col = features.index({torch::indexing::Slice(), i});
                float mean = col.mean().item<float>();
                float std = torch::std(col, false).item<float>();  // unbiased=false
                
                if (std < 1e-6) std = 1.0f;
                
                feature_means[i] = mean;
                feature_stds[i] = std;
            }
        }
        
        for (int i = 0; i < features.size(1); ++i) {
            if (feature_stds[i] > 1e-6) {
                features.index({torch::indexing::Slice(), i}) = 
                    (features.index({torch::indexing::Slice(), i}) - feature_means[i]) / feature_stds[i];
            }
        }
    }
    
    void saveEncodings(const std::string& path) const {
        std::ofstream file(path);
        
        // Save feature normalization stats
        file << "NORMALIZATION " << feature_means.size() << "\n";
        for (size_t i = 0; i < feature_means.size(); ++i) {
            file << feature_means[i] << " " << feature_stds[i] << "\n";
        }
        
        // Save category encoding
        file << "CATEGORY " << category_encoding.size() << "\n";
        for (const auto& [key, val] : category_encoding) {
            file << key << " " << val << "\n";
        }
        
        // Save abc encoding
        file << "ABC " << abc_encoding.size() << "\n";
        for (const auto& [key, val] : abc_encoding) {
            file << key << " " << val << "\n";
        }
        
        // Save status encoding
        file << "STATUS " << status_encoding.size() << "\n";
        for (const auto& [key, val] : status_encoding) {
            file << key << " " << val << "\n";
        }
        
        // Save warehouse encoding
        file << "WAREHOUSE " << warehouse_encoding.size() << "\n";
        for (const auto& [key, val] : warehouse_encoding) {
            file << key << " " << val << "\n";
        }
        
        file.close();
    }
    
    void loadEncodings(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return;
        
        std::string line;
        std::string current_section;
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            ss >> token;
            
            if (token == "NORMALIZATION" || token == "CATEGORY" || token == "ABC" || token == "STATUS" || token == "WAREHOUSE") {
                current_section = token;
                int count;
                ss >> count;
                
                if (current_section == "NORMALIZATION") {
                    feature_means.clear();
                    feature_stds.clear();
                    for (int i = 0; i < count; ++i) {
                        std::getline(file, line);
                        std::stringstream ss_norm(line);
                        float mean, std_val;
                        ss_norm >> mean >> std_val;
                        feature_means.push_back(mean);
                        feature_stds.push_back(std_val);
                    }
                }
            } else {
                float val;
                ss >> val;
                
                if (current_section == "CATEGORY") {
                    category_encoding[token] = (int)val;
                } else if (current_section == "ABC") {
                    abc_encoding[token] = (int)val;
                } else if (current_section == "STATUS") {
                    status_encoding[token] = (int)val;
                } else if (current_section == "WAREHOUSE") {
                    warehouse_encoding[token] = (int)val;
                }
            }
        }
        
        file.close();
    }
};
