# Retail-Demand-Forecasting-System
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/f979ff00-abdc-476c-bc32-7d65ad83f410" />
<br>
<br>
<br>
<br>
### Overview
A comprehensive demand forecasting system that processes historical sales data coming from an E-commerce platform for a retail business, trains machine learning models, and generates future predictions. The pipeline handles the complete workflow from data ingestion to forecast generation.
<br>
<br>
### Project Structure
demand-forecasting/
├── corelib/<br>                   
│   ├── forecast_generator.py <br> 
│   ├── model_trainer.py    <br>   
│   ├── visualizer.py  <br>        
│   └── __init__.py  <br>          
├── config/<br>
│   └── config.json <br>          
├── data/       <br>             
│   ├── input/ <br>              
│   │   ├── sales_data.csv <br>   
│   │   └── holidays.xlsx <br>    
│   └── output/    <br>          
│       └── forecasts.csv   <br>  
├── logs/    <br>                 
├── main.py  <br>                 
└── README.md   <br>    
<br>
<br>
<br>
<br>
### Installation
#### Clone repository
git clone https://github.com/yourusername/demand-forecasting.git<br>
cd demand-forecasting<br>
<br>
#### Create virtual environment
python -m venv venv<br>
source venv/bin/activate  # Linux/Mac<br>
venv\Scripts\activate  # Windows<br>
<br>
#### Install dependencies
pip install -r requirements.txt

