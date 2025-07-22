from datetime import timedelta
import pandas as pd


class ForecastGenerator:
    """Generates forecasts using the best model"""

    def __init__(self, encoders):
        self.encoders = encoders

    def generate_forecasts(self, model, df, last_date, holidays, horizon=7):
        """Generate forecasts for all product-warehouse combinations"""
        logger.info("Generating forecasts...")
        groups = df[['product_id', 'warehouse_id']].drop_duplicates()
        forecasts = []
        holiday_dates = holidays['day'].tolist()

        for idx, (product, warehouse) in groups.iterrows():
            try:
                group_data = df[(df['product_id'] == product) &
                                (df['warehouse_id'] == warehouse)]

                # Get last available data points
                window = group_data['quantity'].tail(7).tolist()
                last_holiday = group_data[group_data['is_holiday'] == 1]['day'].max()

                for days_ahead in range(1, horizon + 1):
                    current_date = last_date + timedelta(days=days_ahead)

                    # Calculate holiday features
                    is_holiday = 1 if current_date in holiday_dates else 0
                    holiday_type = self.encoders['holiday'].transform(
                        [holidays[holidays['day'] == current_date]['Holiday'].values[0]]
                    )[0] if is_holiday else 0

                    days_since = (current_date - last_holiday).days if last_holiday else 0

                    # Create time features
                    adjusted_day_of_week = (current_date.weekday() + 1) % 7  # Sunday=0
                    is_weekend = 1 if adjusted_day_of_week in [0, 6] else 0

                    # Build feature vector
                    features = pd.DataFrame({
                        'product_id': [product],
                        'warehouse_id': [warehouse],
                        'lag1': [window[-1] if window else 0],
                        'lag7': [group_data[group_data['day'] == current_date - timedelta(days=7)]['quantity'].values[0]
                                 if not group_data[group_data['day'] == current_date - timedelta(days=7)].empty else 0,
                                 'rolling_mean7': [np.mean(window) if window else 0],
                    'day_of_week': [adjusted_day_of_week],
                    'month': [current_date.month],
                    'is_weekend': [is_weekend],
                    'is_holiday': [is_holiday],
                    'holiday_type': [holiday_type],
                    'days_since_holiday': [days_since]
                    })

                    # Apply categorical encoding
                    features['product_id'] = self.encoders['product'].transform(features['product_id'])
                    features['warehouse_id'] = self.encoders['warehouse'].transform(features['warehouse_id'])
                    categoricals = ['day_of_week', 'month', 'holiday_type']
                    features[categoricals] = features[categoricals].astype('category')

                    # Make prediction
                    pred = max(0, model.predict(features)[0])
                    forecasts.append({
                        'day': current_date,
                        'warehouse_id': self.encoders['warehouse'].inverse_transform([warehouse])[0],
                        'product_id': self.encoders['product'].inverse_transform([product])[0],
                        'predicted_quantity': pred
                    })

                    # Update rolling window
                    window = window[1:] + [pred] if len(window) >= 7 else window + [pred]

            except Exception as e:
                logger.error(f"Error forecasting for product {product}, warehouse {warehouse}: {e}")

        return pd.DataFrame(forecasts)