import argparse
import logging
import sys
from pathlib import Path
import json

from corelib.data_processor import DataProcessor
from corelib.visualizer import Visualizer
from corelib.model_trainer import ModelTrainer
from corelib.forecast_generator import ForecastGenerator

def load_config():
    config_path = Path(__file__).parent / 'config' / 'config.json'
    with open(config_path) as f:
        return json.load(f)

def setup_logging():
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main(data_path, holidays_path, output_path):
    logger = logging.getLogger(__name__)
    config = load_config()
    
    try:
        # Data processing
        processor = DataProcessor(data_path, holidays_path)
        df = processor.load_and_preprocess()
        
        # Visualization
        visualizer = Visualizer(config)
        visualizer.create_visualizations(df)
        
        # Model training
        trainer = ModelTrainer(config)
        X_train, X_test, y_train, y_test = temporal_split(df)
        results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
        print_metrics(results)
        
        # Forecasting
        forecast_generator = ForecastGenerator(processor.encoders)
        forecast_df = forecast_generator.generate_forecasts(
            best_model(results), df, df['day'].max(), 
            pd.read_excel(holidays_path)
        )
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"Forecasts saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Demand Forecasting Pipeline")
    parser.add_argument('--data', type=str, required=True,
                      help='Path to demand data CSV')
    parser.add_argument('--holidays', type=str, required=True,
                      help='Path to holidays Excel file')
    parser.add_argument('--output', type=str, default='data/output/Forcasted_Demand.csv',
                      help='Output path for forecasts')
    
    args = parser.parse_args()
    
    main(
        data_path=args.data,
        holidays_path=args.holidays,
        output_path=args.output
    )