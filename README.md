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

## Key Features
- **Comprehensive Dataset Generation**: Synthetic soil health data with 25+ features
- **Multiple ML Models**: SVM, ANN, and clustering algorithms
- **Real-time Prediction**: Soil health classification and regression
- **Intelligent Recommendations**: Automated fertilization and irrigation advice
- **Advanced Visualizations**: Interactive dashboards and detailed analytics
- **Performance Evaluation**: Comprehensive model comparison and validation
- **Feature Importance Analysis**: Identification of critical soil parameters

## Project Structure
```
├── data/
│   ├── soil_health_dataset.csv          # Generated synthetic dataset (5,000 samples)
│   ├── soil_health_test_dataset.csv     # Test subset (500 samples)
│   ├── generate_dataset.py              # Alternative dataset generator
│   └── data_description.md              # Comprehensive dataset documentation
├── src/
│   ├── data_generation.py               # Advanced dataset generation
│   ├── data_preprocessing.py            # Data cleaning and preprocessing
│   ├── evaluation.py                    # Model evaluation and metrics
│   ├── visualization.py                 # Comprehensive visualization tools
│   └── models/
│       ├── svm_model.py                 # Support Vector Machine model
│       ├── ann_model.py                 # Artificial Neural Network model
│       └── clustering_model.py          # Clustering algorithms (K-means, DBSCAN, etc.)
├── notebooks/
│   ├── dataset_exploration.ipynb        # Comprehensive EDA
│   ├── model_training.ipynb             # Model training and validation
│   └── results_analysis.ipynb           # Results analysis and insights
├── docs/
│   ├── dataset_summary.md               # Dataset overview
│   └── problem_statement.md             # Detailed problem statement
├── models/                              # Saved trained models (generated)
├── results/                             # Analysis results (generated)
│   └── visualizations/                  # Generated plots and charts
├── main.py                              # Main project runner
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Quick Start

### Option 1: Run Complete Project (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd Edunet_AIML

# Install dependencies
pip install -r requirements.txt

# Run the complete project pipeline
python main.py

# Or run specific steps
python main.py --step data      # Generate dataset only
python main.py --step train     # Train models only
python main.py --step evaluate  # Evaluate models only
```

### Option 2: Step-by-Step Execution
```bash
# 1. Generate dataset
python src/data_generation.py

# 2. Run Jupyter notebooks in order
jupyter notebook notebooks/dataset_exploration.ipynb
jupyter notebook notebooks/model_training.ipynb
jupyter notebook notebooks/results_analysis.ipynb

# 3. Generate visualizations
python -c "from src.visualization import SoilHealthVisualizer; import pandas as pd; viz = SoilHealthVisualizer(pd.read_csv('data/soil_health_dataset.csv')); viz.export_visualizations()"
```

## Model Performance

### Classification Results
- **SVM Classifier**: Accuracy ~95%+, F1-Score ~94%+
- **ANN Classifier**: Accuracy ~93%+, F1-Score ~92%+

### Regression Results
- **SVM Regressor**: R² Score ~0.85+, RMSE ~0.12
- **ANN Regressor**: R² Score ~0.83+, RMSE ~0.13

### Clustering Analysis
- **K-means**: 4 distinct soil condition clusters identified
- **Silhouette Score**: ~0.7+ indicating good cluster separation

## Key Improvisations

### 1. Advanced Dataset Generation
- **Realistic Statistical Modeling**: Used Dirichlet, Gamma, and Lognormal distributions
- **Complex Correlations**: Season-latitude temperature correlations, organic matter-nitrogen relationships
- **Comprehensive Features**: 25+ features including satellite vegetation indices, weather data

### 2. Sophisticated Model Architecture
- **Hyperparameter Optimization**: Grid search and cross-validation
- **Multiple Algorithms**: SVM, ANN, and various clustering methods
- **Ensemble Capabilities**: Framework for model comparison and selection

### 3. Comprehensive Evaluation Framework
- **Multi-metric Assessment**: Accuracy, precision, recall, F1-score, R², RMSE
- **Cross-validation**: Robust model validation with stratified sampling
- **Feature Importance**: Permutation importance and SHAP values

### 4. Interactive Visualizations
- **Plotly Dashboards**: Interactive 3D visualizations and real-time analysis
- **Comprehensive Charts**: PCA analysis, correlation heatmaps, seasonal patterns
- **Export Capabilities**: Automated generation of publication-ready figures

### 5. Production-Ready Pipeline
- **Automated Workflow**: Complete project execution with single command
- **Logging System**: Comprehensive logging for debugging and monitoring
- **Modular Design**: Reusable components for different agricultural applications

## Technical Requirements

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
jupyter>=1.0.0
```

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- 2GB+ free disk space

## Usage Examples

### Predict Soil Health for New Data
```python
from src.models.svm_model import SoilHealthSVM
import pandas as pd

# Load trained model
model = SoilHealthSVM(task_type='classification')
model.load_model('models/svm_classifier')

# Prepare new soil data
new_data = pd.DataFrame({
    'soil_ph': [6.5],
    'soil_moisture': [25.0],
    'nitrogen_content': [45.0],
    # ... other features
})

# Make prediction
prediction = model.predict(new_data)
print(f"Soil health category: {prediction[0]}")
```

### Generate Custom Visualizations
```python
from src.visualization import SoilHealthVisualizer
import pandas as pd

# Load data and create visualizations
df = pd.read_csv('data/soil_health_dataset.csv')
visualizer = SoilHealthVisualizer(df)

# Create specific analysis
visualizer.plot_seasonal_analysis()
visualizer.plot_correlation_analysis()
visualizer.create_interactive_dashboard()
```

## Results and Insights

### Key Findings
1. **pH and moisture content** are the most critical factors for soil health
2. **Seasonal variations** significantly impact soil parameters
3. **Nutrient balance** (NPK) is crucial for optimal soil conditions
4. **Organic matter content** strongly correlates with overall soil health

### Farmer Recommendations
1. **Regular Monitoring**: Implement monthly soil testing for pH and nutrients
2. **Seasonal Adjustments**: Modify irrigation based on seasonal moisture patterns
3. **Precision Fertilization**: Use model predictions for targeted nutrient application
4. **Organic Matter Management**: Maintain organic content above 3% for optimal health

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- AICTE Edunet Foundation for project guidance
- Agricultural science literature for domain knowledge
- Open-source machine learning community for tools and frameworks

## Contact
For questions, suggestions, or collaborations, please reach out through the repository issues or contact the development team.

---

**Note**: This project uses synthetic data for demonstration purposes. For production use, integrate with real soil sensor data and field measurements.

## Usage
See individual notebooks and scripts for detailed usage instructions.

## Contributing
Please read the problem statement and follow the project structure when contributing.
