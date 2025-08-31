# Soil Health Dataset Summary

## Dataset Overview
This synthetic soil health monitoring dataset contains **5,000 samples** of comprehensive soil measurements from agricultural fields across different regions, seasons, and crop types.

## Dataset Files
- `soil_health_dataset.csv` - Main dataset (5,000 samples)
- `soil_health_test_dataset.csv` - Test subset (500 samples)
- `generate_dataset.py` - Script to generate the dataset
- `data_description.md` - Detailed feature descriptions

## Dataset Features (21 columns)

### Identifiers and Context
1. **field_id** - Unique field identifier (F0001 to F5000)
2. **measurement_date** - Date of measurement (2023-09-01 to 2025-08-31)
3. **soil_type** - Type of soil (Clay, Sandy, Loam, Silt, Peat)
4. **crop_type** - Crop being grown (Wheat, Corn, Rice, Soybean, Barley, Cotton, Tomato, Potato)
5. **region** - Geographic region (North, South, East, West, Central)
6. **season** - Season of measurement (Spring, Summer, Fall, Winter)

### Core Soil Properties
7. **soil_moisture_percent** - Soil moisture content (5-80%)
8. **soil_temperature_celsius** - Soil temperature (-5 to 45°C)
9. **ph_level** - Soil pH level (4.0-9.0)
10. **nitrogen_ppm** - Nitrogen content in parts per million (10-100 ppm)
11. **phosphorus_ppm** - Phosphorus content (5-60 ppm)
12. **potassium_ppm** - Potassium content (80-350 ppm)
13. **organic_matter_percent** - Organic matter percentage (0.5-20%)
14. **electrical_conductivity_ds_per_m** - Soil salinity measure (0.1-8.0 dS/m)
15. **bulk_density_g_per_cm3** - Soil compaction measure (0.5-2.0 g/cm³)

### Environmental Data
16. **rainfall_last_week_mm** - Recent rainfall (0-50+ mm)
17. **avg_humidity_percent** - Average humidity (30-95%)

### Vegetation Index
18. **ndvi_vegetation_index** - Normalized Difference Vegetation Index (0-1)

### Target Variables
19. **soil_health_score** - Computed soil health score (0-1, continuous)
20. **health_category** - Categorical health rating (Poor, Fair, Good, Excellent)
21. **recommendations** - Automated management recommendations

## Data Distribution

### Health Categories
- **Good**: 2,656 samples (53.1%)
- **Excellent**: 2,148 samples (43.0%)
- **Fair**: 196 samples (3.9%)
- **Poor**: 0 samples (0.0%)

### Soil Types (evenly distributed)
- Silt: 1,028 samples
- Sandy: 1,018 samples
- Peat: 1,011 samples
- Clay: 993 samples
- Loam: 950 samples

### Crop Types (evenly distributed)
- Rice: 679 samples
- Corn: 664 samples
- Wheat: 623 samples
- Cotton: 617 samples
- Tomato: 616 samples
- Potato: 610 samples
- Barley: 601 samples
- Soybean: 590 samples

## Key Characteristics

### Realistic Relationships
- Soil properties vary realistically by soil type
- Seasonal variations in temperature and moisture
- Regional climate differences
- Crop-specific nutrient requirements

### Complex Target Variable
The soil health score is calculated using:
- pH optimization (25% weight)
- Moisture balance (20% weight)
- Nutrient levels (25% weight)
- Organic matter content (20% weight)
- Salinity levels (10% weight)

### Actionable Recommendations
Automated recommendations include:
- pH adjustment (lime/sulfur application)
- Nutrient supplementation (NPK fertilizers)
- Irrigation management
- Drainage improvement
- Organic matter enhancement
- Salinity reduction

## Data Quality
- **No missing values**
- **No duplicate records**
- **Realistic value ranges**
- **Proper data types**
- **Consistent formatting**

## Potential Use Cases

### Machine Learning Applications
1. **Regression**: Predict continuous soil health scores
2. **Classification**: Categorize soil health levels
3. **Recommendation Systems**: Generate management suggestions
4. **Time Series Analysis**: Track soil health trends
5. **Clustering**: Identify similar soil conditions

### Agricultural Applications
1. **Precision Farming**: Field-specific management
2. **Yield Prediction**: Estimate crop productivity
3. **Resource Optimization**: Efficient fertilizer use
4. **Environmental Monitoring**: Track soil degradation
5. **Decision Support**: Data-driven farming decisions

## Next Steps
1. Explore the dataset using `notebooks/dataset_exploration.ipynb`
2. Develop machine learning models using provided scripts
3. Implement real-time prediction system
4. Create visualization dashboards
5. Deploy for practical use

This comprehensive dataset provides a solid foundation for developing AI-driven soil health monitoring and management solutions.
