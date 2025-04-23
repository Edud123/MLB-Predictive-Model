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
