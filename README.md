# Predicting User Purchase Behavior on an E-commerce Platform

🛒📊🔍🎯

This repository contains Python code for implementing a Logistic Regression model to predict whether a user will buy a product on an e-commerce platform within the next three months. The code covers various stages including data preprocessing, feature engineering, model training, evaluation, and insights generation.

## Purpose
🎯 The purpose of this code is to provide a predictive analytics solution for e-commerce platforms aiming to forecast user purchase behavior. By analyzing historical data and building a predictive model, businesses can gain insights into user preferences, identify potential buyers, and tailor marketing strategies accordingly to improve conversion rates and revenue.

## Usage
1. **Clone the Repository**: Clone this repository to your local machine using the following command:

2. **Install Dependencies**: Ensure you have Python installed on your machine. Install the required dependencies using the following command:

3. **Run the Code**: Execute the Python script `purchase_prediction.py` using your preferred Python environment. This script contains the code for data preprocessing, feature engineering, model training, evaluation, and insights generation.

4. **Input Data**: Ensure you have the input data in CSV format, containing historical user data including features such as user activity, campaign variables, and product purchase history. Modify the `data` variable in the script to point to the location of your input CSV file.

5. **Output**: After running the script, you will obtain model evaluation metrics such as F1 score, precision, recall, ROC-AUC score, and a confusion matrix. Additionally, insights will be generated to understand feature importance and user purchase behavior patterns.

## File Description
- `purchase_prediction.py`: Python script containing the code for predicting user purchase behavior on the e-commerce platform.
- `solution_Kesavan.csv`: CSV file containing the predicted outcomes when applying the trained model to new data.

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Data Preprocessing
- Data cleaning: Missing values are handled by either filling them with a default value or dropping rows/columns.
- Feature engineering: Exploratory data analysis (EDA) is performed to understand the dataset's characteristics. Unique values in each column are examined, and columns with a single unique value are dropped.

## Feature Engineering
- Statistical measures such as mean, median, standard deviation, etc., are calculated to gain insights into the data distribution.
- Relationships between features and the target variable are explored using visualizations such as box plots and correlation matrices.

## Model Training
- The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.
- Feature scaling/normalization is performed to ensure all features contribute equally to the model.
- Logistic regression models are trained on the training data, initially with default hyperparameters and later with optimized hyperparameters using grid search and cross-validation.

## Evaluation
- Model performance is evaluated using various metrics such as F1 score, precision, recall, ROC-AUC score, and confusion matrix.
- The ROC curve is plotted to visualize the trade-off between true positive rate and false positive rate at different threshold levels.
- Hyperparameter tuning aims to improve the model's performance by finding the optimal combination of hyperparameters that maximize the chosen evaluation metric (F1 score).

## Insights Generation
- The EDA reveals insights into the dataset, such as the distribution of target variables, relationships between features, and correlation between variables.
- Feature importance can be inferred from the coefficients of the logistic regression model.
- The model's performance metrics provide an indication of its predictive power and generalization ability on unseen data.

## Contribution
Contributions to this repository are welcome. Feel free to submit pull requests for any improvements or bug fixes.


