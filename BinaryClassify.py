# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')


class BinaryClassification:
    def __init__(self, df, target):
        """Initializes BinaryClassification object for
            Logistic Regression model.

        Args:
            df (pd.DataFrame): Full dataset for use in model
            target (str): Name of the target from DataFrame, must match column string exaclty
        """
        self.df = df.copy()
        self.df.replace(np.nan, 0)
        self.target_label = target
        self.target = np.array(df[target])

    def preprocess_drop(self, ignore_list=[]):
        """Takes a list of columns the user wants to drop from
        from the dataframe before fitting the model. This will
        automatically include the target.

        Args:
            ignore_list (list): list of column names as strings to be removed from dataset
        Returns:
            None: dataframe copy is modified in place
        """
        # append the target to list of columsn to remove
        ignore_list.append(self.target_label)

        # drop each column in ignore_list on the y-axis, modified in-place
        self.df.drop(columns=ignore_list, axis=1, inplace=True)

    def preprocess_encode(self):
        """Uses sklearn LabelEncode to convert categorical data to numeric
            for use in model. 

        Return:
            None : encoding done in place on DataFrame
        """
        # Initialize label encoder object
        label_encode = preprocessing.LabelEncoder()

        # Loop across each column in the dataset
        for col in self.df.columns:
            self.df[col] = label_encode.fit_transform(self.df[col])

    def preprocess_split(self):
        """Takes input dataframe and target array and splits
            into training and validation data using train_test_split
            from sklearn. Function takes no input, and generates the
            split data sets for training and testing.

        Return:
            None: generates train/valid datasets in place on class
        """
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.df, self.target,
                                                                                  train_size=0.7,
                                                                                  test_size=0.3,
                                                                                  random_state=0)

    def preprocess_impute(self):
        """Uses sklearn SimpleImputer() to fill in
            null data. Should be used on predictor dataset
            after being split by train_test_split.

        Return:
            None: training and validation predictors imputed in place
        """
        # Create imputer instance
        dataImpute = SimpleImputer()

        # Run imputer on training dataset, and transform on validation dataset
        self.X_train_imputed = pd.DataFrame(
            dataImpute.fit_transform(self.X_train))
        self.X_valid_imputed = pd.DataFrame(dataImpute.transform(self.X_valid))

        # Add columns back in that are removed in SimpleImputer
        self.X_train_imputed.columns = self.X_train.columns
        self.X_valid_imputed.columns = self.X_valid.columns

    def preprocess(self, ignore_list=[]):
        """Serves as a wrapper for all preprocessing methods. Calls
            preprocessing steps in correct order for model preparation.
            Performs the following:
                1. preprocess_drop: removes columns you don't want to use in the model, removes target
                2. preprocess_encode: convert categorical data into numeric for impute and modeling
                3. preprocess_split: splits dataset into training and validation data
                4. preprocess_impute: impute predictor datasets

        Args:
            ignore_list (list, optional): _description_. Defaults to [].
        """
        self.preprocess_drop(ignore_list)
        self.preprocess_encode()
        self.preprocess_split()
        self.preprocess_impute()

    def fit(self, X, y):
        """This function fits a given set of predictors and target to the model.

        Args:
            X (pd.DataFrame): Input feature predictors for the model
            y (np.ndarry): Array of target for model
        Returns:
            None
        """
        self.model = LogisticRegression(tol=0.0001,
                                        solver='liblinear',
                                        max_iter=5000,
                                        random_state=0,
                                        class_weight={0: 0.2, 1: 0.8})
        self.model.fit(X, y)

    def predict(self, X):
        """Generates prediction from Logistic Regression model
            on a given predictor dataset. Returns ndarray of
            prediction data.

        Args:
            X (pd.DataFrame): Predictor dataset to make a prediction on
        Returns:
            y_pred (np.array): ndarray of prediction data from Logistic Regression
        """

        self.y_pred = self.model.predict(X)

        return(self.y_pred)

    def predict_proba(self, X):
        """Given a predictor dataset, calculates the probability matrix for 
            Logistic Regression model using sklearn predict_proba method.

        Args:
            X (pd.DataFrame): Predictor dataset to predict target probability

        Return:
            y_pred_proba (np.array): Probability maxtrix for the Logistic Regression prediction
        """

        self.y_pred_proba = self.model.predict_proba(X)

        return(self.y_pred_proba)

    def evaluate(self, X, y):
        """Evaluates F1 score and log loss for the Logistic regression
            model.

        Args:
            X (pd.DataFrame): Input predictor dataset
            y (np.ndarray): Ground truth labels as a numpy array of 0-s and 1-s
        Returns:
            eval_dict (dict): dictionary containing f1_score and logloss result
        """
        self.f1_score = f1_score(y, self.y_pred)
        self.logloss = log_loss(y, self.y_pred)

        eval_dict = {'f1_score': self.f1_score, 'logloss': self.logloss}

        return(eval_dict)

    def tune_parameters(self, X, y):
        """Using the predictor DataFrame and ground truth array, will
            return the optimal choice for the following parameters
            maximizing for F1 score in LogisticRegression:
                -solver (possible solvers for LogisticRegression)
                -tol (stopping criteria tolerance)
                -fit_intercept (intercept added to decision function)
                -class_weights (Weights associated with classes in the form {class_label: weight})
                -scores (post-tuning metrics for f1 and logloss)

        Args:
            X (pd.DataFrame): Input predictor dataset
            y (np.ndarray):  Ground truth labels as a numpy array of 0-s and 1-s.
        Returns:
            tuned_params (dict): Output the average scores across all CV validation partitions and best parameters
        """

        # Tuning for optimal LogisticRegression
        #   solver using sklearn cross_val_score
        tuned_params = {}

        solver = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
        solver_score = {}

        for x in solver:

            my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                          ('model', LogisticRegression(solver=x, class_weight='balanced'))])

            scores = cross_val_score(my_pipeline, X, y,
                                     cv=5, scoring='f1')

            solver_score[x] = scores.mean()

        tuned_params['solver'] = max(solver_score, key=solver_score.get)
        # END OF solver tuning

        # Tuning for stopping criteria tolerance
        #   using sklearn cross_val_score
        tolerance = np.linspace(0.0001, 0.1, 10)
        tol_score = {}
        for x in tolerance:

            my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                          ('model', LogisticRegression(tol=x, solver=tuned_params['solver'],
                                                                       class_weight='balanced'))])

            scores = cross_val_score(my_pipeline, X, y,
                                     cv=5, scoring='f1')

            tol_score[x] = scores.mean()

        tuned_params['tol'] = max(tol_score, key=tol_score.get)
        # END OF tol tuning

        # Tuning for decision function fit_intercept
        #   using sklearn cross_val_score
        fit_intercept = [True, False]
        fit_int_score = {}

        for x in fit_intercept:
            my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                          ('model', LogisticRegression(
                                              fit_intercept=x,
                                              tol=tuned_params['tol'],
                                              solver=tuned_params['solver'],
                                              class_weight='balanced'))])

            scores = cross_val_score(my_pipeline, X, y,
                                     cv=5, scoring='f1')

            fit_int_score[x] = scores.mean()

        tuned_params['fit_intercept'] = max(
            fit_int_score, key=fit_int_score.get)
        # END OF fit_intercept tuning

        # Tuning for LogisticRegression class_weight
        #   using sklearn cross_val_score
        class_zero_weights = np.linspace(0.0, 0.99, 50)
        weights_score = {}

        for x in class_zero_weights:
            my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                          ('model', LogisticRegression(
                                              fit_intercept=tuned_params['fit_intercept'],
                                              tol=tuned_params['tol'],
                                              solver=tuned_params['solver'],
                                              class_weight={0: x, 1: 1-x}))])

            scores = cross_val_score(my_pipeline, X, y,
                                     cv=5, scoring='f1')

            weights_score[x] = scores.mean()

        class_zero_weight_max = max(weights_score, key=weights_score.get)

        tuned_params['class_weight'] = {
            0: class_zero_weight_max, 1: 1-class_zero_weight_max}
        # END OF class_weight tuning

        # Scoring metrics f1 and logloss post-tuning
        #   using sklearn cross_val_score
        metrics = ['f1', 'neg_log_loss']
        metrics_score = {}

        for x in metrics:
            my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                          ('model', LogisticRegression(
                                              fit_intercept=tuned_params['fit_intercept'],
                                              tol=tuned_params['tol'],
                                              solver=tuned_params['solver'],
                                              class_weight=tuned_params['class_weight']))])

            scores = cross_val_score(my_pipeline, X, y,
                                     cv=5, scoring=x)

            metrics_score[x] = abs(scores.mean())

            if x == 'f1':
                metrics_score['f1_score'] = metrics_score.pop('f1')
            elif x == 'neg_log_loss':
                metrics_score['logloss'] = metrics_score.pop('neg_log_loss')

        tuned_params['scores'] = metrics_score
        # END OF performance metrics

        return(tuned_params)
