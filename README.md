# Soil Health Monitoring and Management Project

## Project Overview
This project implements an AI-powered soil health monitoring and management system that uses machine learning models to predict soil health conditions and provide actionable insights for farmers.

## Problem Statement
Agricultural productivity and sustainability are increasingly threatened by soil degradation, inefficient fertilization practices, and climate change impacts. Farmers need real-time insights into soil health to make informed decisions about crop management, fertilization, and irrigation.

This project aims to develop a predictive model that:
- Monitors soil health using multiple parameters (moisture, temperature, pH, nutrients)
- Predicts soil health status and crop suitability
- Provides recommendations for optimal fertilization and irrigation
- Helps farmers improve long-term soil vitality and crop yields

## Features
- Synthetic soil health dataset generation
- Multiple ML models (SVM, ANN, Clustering)
- Real-time soil health prediction
- Fertilization recommendations
- Data visualization and analysis
- Performance evaluation metrics

## Project Structure
```
├── data/
│   ├── soil_health_dataset.csv          # Generated synthetic dataset
│   └── data_description.md              # Dataset documentation
├── src/
│   ├── data_generation.py               # Dataset generation script
│   ├── data_preprocessing.py            # Data cleaning and preprocessing
│   ├── models/
│   │   ├── svm_model.py                # Support Vector Machine model
│   │   ├── ann_model.py                # Artificial Neural Network model
│   │   └── clustering_model.py         # Clustering algorithms
│   ├── evaluation.py                   # Model evaluation metrics
│   └── visualization.py                # Data visualization tools
├── notebooks/
│   ├── data_exploration.ipynb          # Exploratory data analysis
│   ├── model_training.ipynb            # Model training and validation
│   └── results_analysis.ipynb          # Results and insights
├── requirements.txt                     # Python dependencies
└── problem_statement.md               # Detailed problem statement
```

## Installation
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run data generation: `python src/data_generation.py`
4. Execute the notebooks in order for complete analysis

## Usage
See individual notebooks and scripts for detailed usage instructions.

## Contributing
Please read the problem statement and follow the project structure when contributing.
