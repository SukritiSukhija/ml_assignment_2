from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
                            n_estimators=300,
                            max_depth=15,
                            random_state=42
                            ),
        "XGBoost":XGBClassifier(
                                n_estimators=300,
                                max_depth=6,
                                learning_rate=0.1,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                eval_metric='logloss',
                                random_state=42
                            )
    }
    return models