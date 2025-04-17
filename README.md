# OrderForecastModel
Problem Statement
UNORG is a fast-growing B2B grocery delivery platform that connects manufacturers, eateries, and retail establishments with essential goods. Our client base spans traditional producers like Petha, Daalmoth, and Revdi makers, as well as modern businesses like restaurants, dhabas, hotels, cafés, and general stores.

As UNORG scales, it becomes increasingly important to anticipate client needs, minimize wastage, and optimize inventory across a highly diverse client base. This project tackles a crucial challenge:

"Which brands should UNORG stock for the next day based on historical customer behavior?"


Solution Overview-

This project implements a brand-level demand prediction pipeline using:
LightGBM classifiers for daily order prediction (per customer)

SMOTE for handling class imbalance

Calibrated probabilities to improve forecast reliability

Auto-regressive features for temporal context (optional)

Expected order quantity estimation using historical averages

Top-N brand recommendation per day based on expected demand and probability

Key Features-

Customer-specific predictions for next-day orders

Brand-level expected quantity forecasts

Dynamic feature engineering using rolling stats and lag

SMOTE-enhanced class balancing for rare purchase events

Hyperparameter tuning via RandomizedSearchCV

Top 150 brands ranked daily by demand likelihood × quantity
