from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, config):
        self.config = config
        self.models = self._initialize_models()

    def _initialize_models(self):
        """Initialize model objects from config"""
        return {
            'LightGBM': lgb.LGBMRegressor(**self.config['models']['LightGBM']),
            'XGBoost': XGBRegressor(**self.config['models']['XGBoost']),
            'CatBoost': CatBoostRegressor(**self.config['models']['CatBoost']),
            'RandomForest': RandomForestRegressor(**self.config['models']['RandomForest'])
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            try:
                model, X_test_processed = self._prepare_model(model, X_train, X_test)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test_processed)
                results[name] = self._calculate_metrics(y_test, y_pred)
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        return results

    def _prepare_model(self, model, X_train, X_test):
        """Handle model-specific data preparation"""
        if isinstance(model, lgb.LGBMRegressor):
            # LightGBM: Specify categorical features
            categoricals = ['product_id', 'warehouse_id',
                            'day_of_week', 'month', 'holiday_type']
            for col in categoricals:
                X_train[col] = X_train[col].astype('category')
                X_test[col] = X_test[col].astype('category')
            return model, X_test

        elif isinstance(model, CatBoostRegressor):
            # CatBoost: Specify categorical indices
            categoricals = ['product_id', 'warehouse_id',
                            'day_of_week', 'month', 'holiday_type']
            return model, X_test

        elif isinstance(model, XGBRegressor):
            # XGBoost: Convert to category dtype
            categoricals = ['day_of_week', 'month', 'holiday_type']
            X_train = X_train.copy()
            X_test = X_test.copy()
            for col in categoricals:
                X_train[col] = X_train[col].astype('category')
                X_test[col] = X_test[col].astype('category')
            return model, X_test

        elif isinstance(model, RandomForestRegressor):
            # RandomForest: Convert categoricals to codes
            X_train = X_train.copy()
            X_test = X_test.copy()
            for col in ['day_of_week', 'month']:
                X_train[col] = X_train[col].cat.codes
                X_test[col] = X_test[col].cat.codes
            return model, X_test

        return model, X_test