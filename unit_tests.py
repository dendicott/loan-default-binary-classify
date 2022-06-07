import unittest
import pandas as pd
import numpy as np
import BinaryClassify as bclf

import warnings
warnings.filterwarnings('ignore')

# list of parameters to remove from model based on preliminary data exploration
params_to_ignore = ['mths_since_last_delinq', 'mths_since_last_record',
                    'open_acc', 'mths_since_last_major_derog', 'initial_list_status',
                    'pymnt_plan', 'collections_12_mths_ex_med']

# URL for dataset
data_url = 'https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Lending_Club_reduced.csv'

# Read dataset into pandas DataFrame
lending_data = pd.read_csv(data_url, index_col='Id')

# Select the target from dataset columns, string must match column label exactly
target = 'is_bad'


class test_binary_classify(unittest.TestCase):
    def test_model_reproducible(self):
        """Test whether model is reporoducbile by 
            creating two instances of the model from the same
            input, and compare the y_pred output for equivalency.
        """
        model1 = bclf.BinaryClassification(lending_data, target)
        model1.preprocess(params_to_ignore)
        model1.fit(model1.X_train_imputed, model1.y_train)
        model1.predict(model1.X_valid_imputed)

        model2 = bclf.BinaryClassification(lending_data, target)
        model2.preprocess(params_to_ignore)
        model2.fit(model2.X_train_imputed, model2.y_train)
        model2.predict(model2.X_valid_imputed)

        self.assertEqual(model1.y_pred.all(),
                         model2.y_pred.all(), "Should be equal")

    def test_missing_values(self):
        """Verify model will still function with missing
            values in the data. Sets a subset of column data
            to NaN, verifies model outputs a logical F1 score.
        """

        lending_data_missing = lending_data.copy()
        lending_data_missing['annual_inc'][0:5] = np.nan

        model1 = bclf.BinaryClassification(lending_data_missing, target)
        model1.preprocess(params_to_ignore)
        model1.fit(model1.X_train_imputed, model1.y_train)
        model1.predict(model1.X_valid_imputed)
        model1.evaluate(model1.X_valid_imputed, model1.y_valid)
        print(len(model1.y_pred))
        model_bool = model1.f1_score > 0 and model1.f1_score < 1
        self.assertEqual(model_bool, True, "Should be True")

    def test_correct_output_format(self):
        """Verify model generates output in the expected format
            np.array per documentation.
        """
        model1 = bclf.BinaryClassification(lending_data, target)
        model1.preprocess(params_to_ignore)
        model1.fit(model1.X_train_imputed, model1.y_train)
        model1.predict(model1.X_valid_imputed)
        expected_output_type = "<class 'numpy.ndarray'>"

        self.assertEqual(str(type(model1.y_pred)), expected_output_type,
                         "Should be equal.")


if __name__ == '__main__':
    unittest.main()
