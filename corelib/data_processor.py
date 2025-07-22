import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering"""

    def __init__(self, data_path, holidays_path):
        self.data_path = data_path
        self.holidays_path = holidays_path
        self.encoders = {
            'product': LabelEncoder(),
            'warehouse': LabelEncoder(),
            'holiday': LabelEncoder()
        }

    def load_and_preprocess(self):
        """Load and preprocess raw data"""
        try:
            logger.info("Loading and preprocessing data...")
            df = pd.read_csv(self.data_path)
            holidays = pd.read_excel(self.holidays_path, parse_dates=['Date'])

            # Basic preprocessing
            df['day'] = pd.to_datetime(df['day'])
            holidays = holidays.rename(columns={'Date': 'day'})[['day', 'Holiday']]

            return self._process_data(df, holidays)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            sys.exit(1)

    def _process_data(self, df, holidays):
        """Internal data processing pipeline"""
        # Generate complete date range
        complete_df = self._create_complete_df(df)

        # Merge data
        df = pd.merge(complete_df, df, on=['day', 'product_id', 'warehouse_id'], how='left')
        df = pd.merge(df, holidays, on='day', how='left')

        # Handle missing values
        df['quantity'] = df['quantity'].fillna(0)
        df['amount'] = df['amount'].fillna(0)

        # Feature engineering
        df = self._create_features(df)

        # Encode categorical variables
        df = self._encode_features(df)

        return df

    def _create_complete_df(self, df):
        """Create complete date-product-warehouse combinations"""
        all_combinations = df[['product_id', 'warehouse_id']].drop_duplicates()
        all_dates = pd.date_range(df['day'].min(), df['day'].max(), freq='D')
        return pd.merge(
            all_combinations.assign(key=1),
            pd.DataFrame({'day': all_dates, 'key': 1}),
            on='key'
        ).drop('key', axis=1)

    def _create_features(self, df):
        """Create temporal and holiday-related features"""
        # Sort for time-based features
        df.sort_values(['product_id', 'warehouse_id', 'day'], inplace=True)

        # Lag features
        df['lag1'] = df.groupby(['product_id', 'warehouse_id'])['quantity'].shift(1)
        df['lag7'] = df.groupby(['product_id', 'warehouse_id'])['quantity'].shift(7)
        df['rolling_mean7'] = df.groupby(['product_id', 'warehouse_id'])['quantity'].transform(
            lambda x: x.rolling(7, min_periods=1).mean().shift(1))

        # Date features
        df['day_of_week'] = ((df['day'].dt.dayofweek + 1) % 7).astype('category')
        df['month'] = df['day'].dt.month.astype('category')
        df['is_weekend'] = df['day_of_week'].isin([0, 6]).astype(int)

        # Holiday features
        df['is_holiday'] = df['Holiday'].notna().astype(int)
        df['days_since_holiday'] = df.groupby(['product_id', 'warehouse_id'])['is_holiday'].transform(
            lambda x: x[::-1].cumsum().shift().fillna(0)[::-1])

        return df.dropna(subset=['lag1', 'lag7', 'rolling_mean7'])

    def _encode_features(self, df):
        """Encode categorical features"""
        df['product_id'] = self.encoders['product'].fit_transform(df['product_id'])
        df['warehouse_id'] = self.encoders['warehouse'].fit_transform(df['warehouse_id'])
        df['holiday_type'] = self.encoders['holiday'].fit_transform(
            df['Holiday'].fillna('No_Holiday')
        )
        return df