import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_soil_health_dataset(num_samples=5000):
    """
    Generate a comprehensive synthetic soil health dataset
    """
    
    # Define soil types and their characteristics
    soil_types = ['Clay', 'Sandy', 'Loam', 'Silt', 'Peat']
    
    # Define crop types
    crop_types = ['Wheat', 'Corn', 'Rice', 'Soybean', 'Barley', 'Cotton', 'Tomato', 'Potato']
    
    # Define regions
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Define seasons
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    
    data = []
    
    for i in range(num_samples):
        # Basic identifiers
        field_id = f"F{i+1:04d}"
        soil_type = random.choice(soil_types)
        crop_type = random.choice(crop_types)
        region = random.choice(regions)
        season = random.choice(seasons)
        
        # Generate date (last 2 years)
        start_date = datetime.now() - timedelta(days=730)
        random_days = random.randint(0, 730)
        measurement_date = start_date + timedelta(days=random_days)
        
        # Soil moisture (%) - varies by soil type and season
        base_moisture = {
            'Clay': 35, 'Sandy': 15, 'Loam': 25, 'Silt': 30, 'Peat': 60
        }
        seasonal_modifier = {
            'Spring': 1.2, 'Summer': 0.7, 'Fall': 1.0, 'Winter': 1.3
        }
        moisture = base_moisture[soil_type] * seasonal_modifier[season]
        moisture += np.random.normal(0, 3)  # Add noise
        moisture = max(5, min(80, moisture))  # Clamp between 5-80%
        
        # Soil temperature (°C) - varies by season and region
        base_temp = {
            'Spring': 15, 'Summer': 25, 'Fall': 12, 'Winter': 5
        }
        regional_modifier = {
            'North': -3, 'South': 5, 'East': 0, 'West': 2, 'Central': 0
        }
        temperature = base_temp[season] + regional_modifier[region]
        temperature += np.random.normal(0, 2)
        temperature = max(-5, min(45, temperature))
        
        # pH level (4.0-9.0) - varies by soil type
        base_ph = {
            'Clay': 7.2, 'Sandy': 6.0, 'Loam': 6.8, 'Silt': 7.0, 'Peat': 5.5
        }
        ph = base_ph[soil_type] + np.random.normal(0, 0.3)
        ph = max(4.0, min(9.0, ph))
        
        # Nitrogen content (ppm) - affected by crop type and season
        base_nitrogen = {
            'Wheat': 45, 'Corn': 55, 'Rice': 40, 'Soybean': 35, 
            'Barley': 42, 'Cotton': 50, 'Tomato': 60, 'Potato': 48
        }
        nitrogen = base_nitrogen[crop_type]
        if season in ['Spring', 'Summer']:
            nitrogen *= 1.2  # Higher during growing season
        nitrogen += np.random.normal(0, 8)
        nitrogen = max(10, min(100, nitrogen))
        
        # Phosphorus content (ppm)
        phosphorus = np.random.normal(25, 8)
        phosphorus = max(5, min(60, phosphorus))
        
        # Potassium content (ppm)
        potassium = np.random.normal(180, 40)
        potassium = max(80, min(350, potassium))
        
        # Organic matter (%)
        base_organic = {
            'Clay': 3.5, 'Sandy': 1.8, 'Loam': 4.2, 'Silt': 3.0, 'Peat': 12.0
        }
        organic_matter = base_organic[soil_type] + np.random.normal(0, 0.5)
        organic_matter = max(0.5, min(20, organic_matter))
        
        # Electrical conductivity (dS/m) - salinity measure
        ec = np.random.gamma(2, 0.5)
        ec = max(0.1, min(8.0, ec))
        
        # Bulk density (g/cm³)
        base_density = {
            'Clay': 1.3, 'Sandy': 1.6, 'Loam': 1.4, 'Silt': 1.35, 'Peat': 0.8
        }
        bulk_density = base_density[soil_type] + np.random.normal(0, 0.1)
        bulk_density = max(0.5, min(2.0, bulk_density))
        
        # Weather data
        rainfall_last_week = max(0, np.random.gamma(2, 5))  # mm
        avg_humidity = np.random.normal(65, 15)  # %
        avg_humidity = max(30, min(95, avg_humidity))
        
        # Satellite vegetation index (NDVI) - 0 to 1
        base_ndvi = 0.6 if season in ['Spring', 'Summer'] else 0.3
        ndvi = base_ndvi + np.random.normal(0, 0.15)
        ndvi = max(0, min(1, ndvi))
        
        # Calculate soil health score (target variable)
        # This is a composite score based on multiple factors
        ph_score = 1 - abs(ph - 6.8) / 3.2  # Optimal pH around 6.8
        moisture_score = 1 - abs(moisture - 30) / 50  # Optimal moisture around 30%
        nutrient_score = (
            min(nitrogen / 50, 1) * 0.4 +  # Nitrogen contribution
            min(phosphorus / 30, 1) * 0.3 +  # Phosphorus contribution
            min(potassium / 200, 1) * 0.3   # Potassium contribution
        )
        organic_score = min(organic_matter / 5, 1)  # Optimal organic matter >= 5%
        salinity_score = max(0, 1 - ec / 4)  # Lower salinity is better
        
        # Weighted soil health score
        soil_health_score = (
            ph_score * 0.25 +
            moisture_score * 0.20 +
            nutrient_score * 0.25 +
            organic_score * 0.20 +
            salinity_score * 0.10
        )
        
        # Add some randomness and clamp between 0-1
        soil_health_score += np.random.normal(0, 0.05)
        soil_health_score = max(0, min(1, soil_health_score))
        
        # Determine health category
        if soil_health_score >= 0.8:
            health_category = 'Excellent'
        elif soil_health_score >= 0.6:
            health_category = 'Good'
        elif soil_health_score >= 0.4:
            health_category = 'Fair'
        else:
            health_category = 'Poor'
        
        # Recommendations based on soil conditions
        recommendations = []
        if ph < 6.0:
            recommendations.append('Add lime to increase pH')
        elif ph > 7.5:
            recommendations.append('Add sulfur to decrease pH')
        
        if nitrogen < 30:
            recommendations.append('Apply nitrogen fertilizer')
        if phosphorus < 15:
            recommendations.append('Apply phosphorus fertilizer')
        if potassium < 120:
            recommendations.append('Apply potassium fertilizer')
        
        if moisture < 20:
            recommendations.append('Increase irrigation')
        elif moisture > 50:
            recommendations.append('Improve drainage')
        
        if organic_matter < 2:
            recommendations.append('Add organic compost')
        
        if ec > 2:
            recommendations.append('Reduce soil salinity')
        
        if not recommendations:
            recommendations.append('Maintain current practices')
        
        # Compile data point
        data_point = {
            'field_id': field_id,
            'measurement_date': measurement_date.strftime('%Y-%m-%d'),
            'soil_type': soil_type,
            'crop_type': crop_type,
            'region': region,
            'season': season,
            'soil_moisture_percent': round(moisture, 2),
            'soil_temperature_celsius': round(temperature, 2),
            'ph_level': round(ph, 2),
            'nitrogen_ppm': round(nitrogen, 2),
            'phosphorus_ppm': round(phosphorus, 2),
            'potassium_ppm': round(potassium, 2),
            'organic_matter_percent': round(organic_matter, 2),
            'electrical_conductivity_ds_per_m': round(ec, 2),
            'bulk_density_g_per_cm3': round(bulk_density, 3),
            'rainfall_last_week_mm': round(rainfall_last_week, 1),
            'avg_humidity_percent': round(avg_humidity, 1),
            'ndvi_vegetation_index': round(ndvi, 3),
            'soil_health_score': round(soil_health_score, 3),
            'health_category': health_category,
            'recommendations': '; '.join(recommendations)
        }
        
        data.append(data_point)
    
    return pd.DataFrame(data)

def main():
    """
    Generate and save the soil health dataset
    """
    print("Generating soil health monitoring dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate dataset
    df = generate_soil_health_dataset(num_samples=5000)
    
    # Display basic statistics
    print(f"\nDataset generated with {len(df)} samples")
    print(f"Features: {len(df.columns)}")
    print(f"Date range: {df['measurement_date'].min()} to {df['measurement_date'].max()}")
    
    # Save to CSV
    csv_path = 'data/soil_health_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDataset saved to: {csv_path}")
    
    # Display sample data
    print("\nSample of the dataset:")
    print(df.head())
    
    # Display summary statistics
    print("\nNumerical features summary:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe())
    
    # Display categorical features distribution
    print("\nCategorical features distribution:")
    categorical_cols = ['soil_type', 'crop_type', 'region', 'season', 'health_category']
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    # Create a smaller test dataset
    test_df = df.sample(n=500, random_state=42)
    test_path = 'data/soil_health_test_dataset.csv'
    test_df.to_csv(test_path, index=False)
    print(f"\nTest dataset (500 samples) saved to: {test_path}")
    
    return df

if __name__ == "__main__":
    dataset = main()
