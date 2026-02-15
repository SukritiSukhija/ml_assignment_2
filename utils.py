import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(train_path, test_path):
    columns = [
        "age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week",
        "native_country","income"
    ]

    train_df = pd.read_csv(train_path, names=columns, na_values=" ?", skipinitialspace=True)
    test_df = pd.read_csv(test_path, names=columns, na_values=" ?", skipinitialspace=True, skiprows=1)

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    replace_map = {
        ">50K":1, ">50K.":1,
        "<=50K":0, "<=50K.":0
    }

    train_df["income"] = train_df["income"].replace(replace_map)
    test_df["income"] = test_df["income"].replace(replace_map)

    combined = pd.concat([train_df, test_df], axis=0)
    combined = pd.get_dummies(combined, drop_first=True)

    train_encoded = combined.iloc[:len(train_df)]
    test_encoded = combined.iloc[len(train_df):]

    X_train = train_encoded.drop("income", axis=1)
    y_train = train_encoded["income"]

    X_test = test_encoded.drop("income", axis=1)
    y_test = test_encoded["income"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test