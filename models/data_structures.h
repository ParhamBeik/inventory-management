#pragma once

struct InventoryRecord {
    // Identifiers
    std::string sku_id;
    std::string sku_name;
    
    // Categorical features
    std::string category;
    std::string abc_class;
    std::string supplier_id;
    std::string warehouse_id;
    std::string inventory_status;
    std::string fifo_fefo;
    
    // Numerical features
    float stock_age_days;
    int quantity_on_hand;
    int quantity_reserved;
    int quantity_committed;
    int damaged_qty;
    int returns_qty;
    float avg_daily_sales;
    float forecast_next_30d;
    float days_of_inventory;
    int reorder_point;
    int safety_stock;
    int lead_time_days;
    float unit_cost_usd;
    float last_purchase_price_usd;
    float total_inventory_value_usd;
    float sku_churn_rate;
    float order_frequency_per_month;
    float supplier_ontime_pct;
    int count_variance;
    float audit_variance_pct;
    float demand_forecast_accuracy_pct;
    
    // Temporal features (converted from strings)
    int received_date_days;
    int last_purchase_date_days;
    int expiry_date_days;
    int audit_date_days;
    
    // Targets
    float regression_target;
    int classification_target;
};
