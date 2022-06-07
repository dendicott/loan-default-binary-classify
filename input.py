import BinaryClassify as bclf
import pandas as pd
import os

# EXAMPLE USER INPUT BINARY CLASSIFY CLASS
# list of parameters to remove from model based on preliminary data exploration
params_to_ignore = ['mths_since_last_delinq', 'mths_since_last_record',
                    'open_acc', 'mths_since_last_major_derog', 'initial_list_status',
                    'pymnt_plan', 'collections_12_mths_ex_med', 'zip_code', 'inq_last_6mths',
                    'policy_code', 'home_ownership']

# URL for dataset
data_url = 'https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Lending_Club_reduced.csv'

# Read dataset into pandas DataFrame
lending_data = pd.read_csv(data_url, index_col='Id')

# Select the target from dataset columns, string must match column label exactly
target = 'is_bad'

# Initialize the class to creat model
classify = bclf.BinaryClassification(lending_data, target)
# END OF USER INPUT

# Model method calls, run all preprocessing methods on dataset
classify.preprocess(params_to_ignore)
# Fit the imputed X_train and y_train data to the model
classify.fit(classify.X_train_imputed, classify.y_train)
# Use model to make predictions on the imputed X_valid data
predictions = classify.predict(classify.X_valid_imputed)
# Generate probability matrix using the imputed X_valid data
predict_proba = classify.predict_proba(classify.X_valid_imputed)
# Run model performance metrics on the predicted data vs. the validation data
evaluate = classify.evaluate(classify.X_valid_imputed, classify.y_valid)
# Tune the model with K-fold cross validation
tune = classify.tune_parameters(classify.X_train_imputed, classify.y_train)

# Write prediction result to csv
predict_df = pd.DataFrame()
predict_df['Prediction'] = classify.y_pred
predict_df['Validation'] = classify.y_valid

filename = 'model_predictions.csv'
predict_df.to_csv(filename)

# Print all project outputs to terminal
print("Model output array:")
print(predictions)
print('\n')
print("Model probability matrix:")
print(predict_proba)
print('\n')
print("Model evaluation criteria:")
print(evaluate)
print('\n')
print("Model tuning results:")
print(tune)
