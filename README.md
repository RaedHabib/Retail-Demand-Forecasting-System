# Retail-Demand-Forecasting-System
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/f979ff00-abdc-476c-bc32-7d65ad83f410" />
<br>
<br>
<br>
<br>
## Overview
A comprehensive demand forecasting system that processes historical sales data coming from an E-commerce platform for a retail business, trains machine learning models, and generates future predictions. The pipeline handles the complete workflow from data ingestion to forecast generation.
<br>
<br>
## Project Structure
demand-forecasting/
├── corelib/                   
│   ├── forecast_generator.py  
│   ├── model_trainer.py       
│   ├── visualizer.py          
│   └── __init__.py            
├── config/
│   └── config.json           
├── data/                    
│   ├── input/               
│   │   ├── sales_data.csv    
│   │   └── holidays.xlsx     
│   └── output/              
│       └── forecasts.csv     
├── logs/                     
├── main.py                   
└── README.md       
<br>
<br>
<br>
<br>
## Installation
# Clone repository
git clone https://github.com/yourusername/demand-forecasting.git
cd demand-forecasting

### Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
### venv\Scripts\activate  # Windows

### Install dependencies
pip install -r requirements.txt
<br>
<br>
## Usage
### Basic execution:
python main.py \
  --data data/input/sales_data.csv \
  --holidays data/input/holidays.xlsx \
  --output data/output/forecasts.csv


