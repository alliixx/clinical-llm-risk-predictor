# counterfactuals_generator.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import dice_ml
from dice_ml import Dice

# --- Feature definitions ---
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
cat_features = ['sex', 'smoker', 'region']
num_features = ['age', 'bmi', 'children']

def generate_counterfactuals_for_query(query_instance, max_cost=10000):
    # Load and prepare data
    df = pd.read_csv("insurance.csv")
    X = df[features]
    y = df['charges']

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing', add_indicator=True)),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler())
        ]), num_features)
    ])

    # Define and train model
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42))
    ])
    model_pipeline.fit(X, y)

    # Preprocess training data for DiCE
    X_train_transformed = pd.DataFrame(
        preprocessor.transform(X),
        columns=preprocessor.get_feature_names_out()
    )
    X_train_transformed['charges'] = y.reset_index(drop=True)

    data_dice = dice_ml.Data(
        dataframe=X_train_transformed,
        continuous_features=[col for col in X_train_transformed.columns if col != 'charges'],
        outcome_name='charges'
    )

    model_dice = dice_ml.Model(
        model=model_pipeline.named_steps['regressor'],
        backend='sklearn',
        model_type='regressor'
    )

    # Step 2: Generate Counterfactuals
    query_transformed = preprocessor.transform(query_instance)
    query_df = pd.DataFrame(query_transformed, columns=preprocessor.get_feature_names_out())

    explainer = Dice(data_dice, model_dice, method='random')
    dice_exp = explainer.generate_counterfactuals(
        query_df,
        total_CFs=10,
        desired_range=[0, max_cost],
        features_to_vary='all'
    )

    cf_df = dice_exp.cf_examples_list[0].final_cfs_df

    # Step 3: Inverse Transform and Filter
    cat_transformer = preprocessor.named_transformers_['cat']
    num_transformer = preprocessor.named_transformers_['num']

    ohe = cat_transformer.named_steps['encoder']
    cat_columns = ohe.get_feature_names_out(cat_features)
    cat_columns_prefixed = [f'cat__{col}' for col in cat_columns]
    num_columns_prefixed = [f'num__{col}' for col in num_features]

    encoded_cat_df = cf_df[cat_columns_prefixed]
    scaled_num_df = cf_df[num_columns_prefixed]

    recovered_cats = ohe.inverse_transform(encoded_cat_df)
    recovered_cat_df = pd.DataFrame(recovered_cats, columns=cat_features)

    scaler = num_transformer.named_steps['scaler']
    imputer = num_transformer.named_steps['imputer']
    unscaled_num = scaler.inverse_transform(scaled_num_df)
    unscaled_num = imputer.inverse_transform(unscaled_num)
    recovered_num_df = pd.DataFrame(unscaled_num, columns=num_features)

    if 'children' in recovered_num_df.columns:
        recovered_num_df['children'] = recovered_num_df['children'].fillna(0).round().astype(int)

    readable_df = pd.concat([recovered_cat_df, recovered_num_df], axis=1)
    if 'charges' in cf_df.columns:
        readable_df['charges'] = cf_df['charges'].values

    feature_columns = [col for col in readable_df.columns if col != 'charges']
    readable_df = readable_df.drop_duplicates(subset=feature_columns)
    differences = readable_df[features].ne(query_instance.iloc[0]).any(axis=1)
    filtered_df = readable_df[differences].reset_index(drop=True)

    return filtered_df
