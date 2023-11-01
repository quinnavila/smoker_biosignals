import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

categorical = ['hearing(left)', 'hearing(right)', 'dental caries']
numerical = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
             'eyesight(right)', 'systolic', 'relaxation', 'fasting blood sugar', 
             'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 
             'Urine protein', 'serum creatinine', 'AST', 'ALT', 'Gtp']
target = ['smoking']


def training_data():
    df = pd.read_csv('data/train_dataset.csv')

    df['Log_HDL_LDL_Ratio'] = np.log1p(df['HDL'] / df['LDL'])  
    numerical.append('Log_HDL_LDL_Ratio')

    X = df.drop(columns=['smoking'])
    y = df[target].values.ravel() 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, X_val, y_train, y_val, numerical_features, categorical_features, bst_params):
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])

    # Use ColumnTransformer to apply the appropriate transformations to each feature type
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the preprocessing and modeling pipeline for the random forest classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_jobs=-1, random_state=42, **bst_params))
    ])

    # Fit the model to the data
    model.fit(X_train, y_train)

    # Make predictions
    train_preds = model.predict_proba(X_train)[:, 1]
    train_score = roc_auc_score(y_train, train_preds)
    print(f'ROC training score: {round(train_score, 2)}')

    val_preds = model.predict_proba(X_val)[:, 1]

    # Evaluate the model
    val_score = roc_auc_score(y_val, val_preds)
    print(f'ROC validation score: {round(val_score, 4)}')

    return model


def main_training():
    X_train, X_test, y_train, y_test = training_data()
    bst_params = {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 1000}
    model = train_random_forest(X_train, X_test, y_train, y_test, numerical, categorical, bst_params)
    model_filename = 'models/random_forest_model.pkl'
    joblib.dump(model, model_filename, compress=('zlib', 8))
    print(f'Model saved: {model_filename}')

if __name__=="__main__":
    main_training()

# Sample output
# ROC training score: 1.0
# ROC validation score: 0.8907
# Model saved: models/random_forest_model.pkl