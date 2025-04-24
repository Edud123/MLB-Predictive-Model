## How to Run the App

To run the MLB Predictive Model dashboard, follow these steps:

1. **Install Dependencies**:
   - Ensure you have Python installed (version 3.8 or higher).
   - Install the required Python packages by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Train the Models**:
   - Before running the dashboard, ensure the models are trained and saved. Use the `train_models.py` script to train the Random Forest, XGBoost, and Neural Network models:
     ```bash
     python train_models.py
     ```
   - This will generate the necessary model files (`rf_model.pkl`, `xgb_model.pkl`, etc.) in the workspace.

3. **Run the Dashboard**:
   - Start the Streamlit dashboard by running the following command:
     ```bash
     streamlit run mlb_dashboard.py
     ```
   - Open the provided URL in your web browser to interact with the dashboard.

4. **Optional**:
   - If you want to update the data or retrain the models, ensure the CSV files in the workspace are up-to-date and re-run the `train_models.py` script.

### Debugging Steps for `ValueError` in `px.bar`

If you encounter the error `ValueError: Cannot accept list of column references or list of columns for both x and y` while running the dashboard, follow these steps to debug and resolve the issue:

1. **Inspect `category_impact`**:
   - Add a debug statement in the `mlb_dashboard.py` file to check the contents of `category_impact` before passing it to `px.bar`:
     ```python
     st.write("Category Impact Debug:", category_impact)
     ```

2. **Check for Empty Data**:
   - Ensure that `category_impact` is not empty. If it is, the issue likely lies in the logic that calculates it. Review the code that populates `category_impact` to ensure it is correctly aggregating data.

3. **Validate Data Types**:
   - Ensure that `list(category_impact.keys())` and `list(category_impact.values())` are valid lists of strings and numbers, respectively. If not, inspect the data source (`filtered_df`) and the correlation calculation logic.

4. **Handle Empty Data Gracefully**:
   - Add a fallback in the `mlb_dashboard.py` file to handle cases where `category_impact` is empty:
     ```python
     if not category_impact:
         st.warning("No data available for category impact analysis.")
         return
     ```

5. **Test with Sample Data**:
   - If the issue persists, test the `px.bar` function with sample data to isolate the problem. For example:
     ```python
     fig_impact = px.bar(
         x=["Category A", "Category B"],
         y=[0.5, 0.7],
         title="Sample Data Test"
     )
     ```

By following these steps, you should be able to identify and resolve the root cause of the error.

---

# MLB-Predictive-Model
Final Project for Data Analytics Class at CBS

## Project Overview
This project implements machine learning models to predict MLB team win rates based on historical performance metrics from 2010-2025. The models utilize batting, pitching, and fielding statistics with a focus on 5-year lagged features.

## Models Implemented
1. Random Forest Regressor
2. XGBoost Regressor 
3. Neural Network

## Data Analysis

### Features
- Batting metrics: wRC+, OBP, SLG, HR, WAR, Off, Def, BsR, K%, BB%
- Pitching metrics: ERA, FIP, xFIP, xERA, K/9, BB/9, HR/9, LOB%, GB%, WAR
- Fielding metrics: Def, DRS, OAA, UZR, UZR/150, RngR, ErrR, ARM, FRM, DPR
- 5-year lag features for each metric

### Model Performance
- Random Forest R² Score: ~0.33
- XGBoost R² Score: ~0.33 
- Neural Network: Underperformed compared to tree-based models

## Key Findings

### Strengths
1. Comprehensive feature set covering all aspects of baseball performance
2. Multiple modeling approaches allowing for comparison
3. Strong feature importance analysis revealing ERA_lag1 as key predictor
4. Good handling of temporal dependencies through lag features

### Limitations

1. Data Quality Issues:
- Missing values in multiple features
- Limited sample size for neural network training
- Potential data leakage in validation approach

2. Modeling Approach:
- No cross-validation implemented
- Limited hyperparameter tuning
- Neural network architecture may not be optimal
- No ensemble methods explored

3. Feature Engineering:
- Lack of interaction terms
- No feature selection process
- Missing important contextual features

## Recommendations for Improvement

### Data Preprocessing
1. Implement proper cross-validation with time-based splits
2. Handle missing values more systematically
3. Add feature selection process
4. Consider dimensionality reduction techniques

### Feature Engineering
1. Create interaction terms between key metrics
2. Add contextual features:
   - Home/Away game splits
   - Injury data
   - Weather conditions
   - Travel distance
   - Rest days
   - Roster changes
   - Manager changes

### Model Enhancements
1. Implement proper hyperparameter tuning:
   - Grid search or Bayesian optimization
   - Cross-validation for parameter selection
2. Try ensemble methods:
   - Stacking different models
   - Weighted averaging of predictions
3. Improve neural network:
   - Add dropout layers
   - Try different architectures
   - Implement batch normalization
   - Use learning rate scheduling

### Validation
1. Implement proper time-series cross-validation
2. Add confidence intervals for predictions
3. Create more sophisticated evaluation metrics:
   - MAPE (Mean Absolute Percentage Error)
   - Custom baseball-specific metrics

### Additional Features
1. Add player-level data aggregation
2. Include market/payroll data
3. Add strength of schedule metrics
4. Include momentum indicators

## Future Work
1. Implement real-time prediction updates
2. Add visualization dashboard
3. Create player contribution analysis
4. Develop scenario analysis tools

## Technical Debt
1. Need proper documentation for functions
2. Missing error handling
3. No logging implementation
4. Code organization could be improved
5. Missing unit tests
