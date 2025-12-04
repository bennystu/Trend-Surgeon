from xgboost import XGBRegressor
import joblib
from pathlib import Path


class XGBoostModel:
    """
    Wrapper for XGBoost regression model used in Trend-Surgeon.
    Only handles:
        - initialize with params
        - train
        - predict
        - save and load
    The dataset building and the train/validation split happen outside this file.
    """

    def __init__(self, params=None):
        """
        params: dict of hyperparameters for XGBRegressor
        """
        if params is None:
            params = {
                "learning_rate": 0.1,
                "n_estimators": 200,
                "max_depth": 3,
                "subsample": 1,
                "objective": "reg:squarederror",
                "tree_method": "hist"
            }

        self.params = params
        self.model = XGBRegressor(**params)

    def train(self, X_train, y_train):
        """Train the model on training data."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict future values."""
        return self.model.predict(X)

    def save(self, path: str):
        """Save trained model to disk."""
        path = Path(path)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        """Load a saved model from disk."""
        path = Path(path)
        loaded_model = joblib.load(path)

        # Create wrapper instance
        instance = cls()
        instance.model = loaded_model
        return instance
