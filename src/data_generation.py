"""
Soil Health Dataset Generation Script

This script generates a synthetic dataset for soil health monitoring and management.
The dataset includes various soil parameters, environmental factors, and target variables
for training machine learning models.

"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class SoilDataGenerator:
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.data = {}
        
    def generate_location_data(self):
        """Generate geographical and location-based features"""
        # Latitude and Longitude (focusing on agricultural regions)
        self.data['latitude'] = np.random.uniform(25.0, 45.0, self.n_samples)  # Northern agricultural regions
        self.data['longitude'] = np.random.uniform(-120.0, -70.0, self.n_samples)  # North American longitude
        
        # Elevation (meters above sea level)
        self.data['elevation'] = np.random.gamma(2, 150, self.n_samples)  # Gamma distribution for realistic elevation
        self.data['elevation'] = np.clip(self.data['elevation'], 0, 3000)
        
    def generate_temporal_data(self):
        """Generate time-based features"""
        # Random dates over 2 years
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        dates = []
        for _ in range(self.n_samples):
            random_days = random.randint(0, (end_date - start_date).days)
            random_date = start_date + timedelta(days=random_days)
            dates.append(random_date)
        
        self.data['date'] = dates
        self.data['month'] = [d.month for d in dates]
        self.data['season'] = [self._get_season(d.month) for d in dates]
        
    def _get_season(self, month):
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def generate_soil_physical_properties(self):
        """Generate soil physical properties"""
        # Soil moisture content (%) - varies by season and location
        base_moisture = np.random.normal(25, 8, self.n_samples)
        seasonal_adjustment = [5 if s in ['Spring', 'Winter'] else -3 for s in self.data['season']]
        self.data['soil_moisture'] = np.clip(base_moisture + seasonal_adjustment, 5, 50)
        
        # Soil temperature (°C) - correlated with season and latitude
        base_temp = 15 + (45 - self.data['latitude']) * 0.5  # Warmer at lower latitudes
        seasonal_temp_adj = []
        for season in self.data['season']:
            if season == 'Summer':
                adj = np.random.normal(8, 2)
            elif season == 'Winter':
                adj = np.random.normal(-8, 3)
            elif season == 'Spring':
                adj = np.random.normal(2, 2)
            else:  # Fall
                adj = np.random.normal(-2, 2)
            seasonal_temp_adj.append(adj)
        
        self.data['soil_temperature'] = np.clip(base_temp + seasonal_temp_adj, -5, 35)
        
        # Soil pH (4.0 - 9.0, with most soils being slightly acidic to neutral)
        self.data['soil_ph'] = np.random.gamma(2, 1.5, self.n_samples) + 4.5
        self.data['soil_ph'] = np.clip(self.data['soil_ph'], 4.0, 9.0)
        
        # Bulk density (g/cm³)
        self.data['bulk_density'] = np.random.normal(1.3, 0.2, self.n_samples)
        self.data['bulk_density'] = np.clip(self.data['bulk_density'], 0.8, 1.8)
        
        # Soil texture - Sand, Silt, Clay percentages (must sum to 100)
        # Generate using Dirichlet distribution for realistic soil texture
        texture_samples = np.random.dirichlet([3, 2, 1], self.n_samples) * 100
        self.data['sand_percentage'] = texture_samples[:, 0]
        self.data['silt_percentage'] = texture_samples[:, 1]
        self.data['clay_percentage'] = texture_samples[:, 2]
        
    def generate_soil_chemical_properties(self):
        """Generate soil chemical properties"""
        # Nitrogen content (mg/kg) - affected by fertilization and organic matter
        self.data['nitrogen_content'] = np.random.lognormal(3, 0.5, self.n_samples)
        self.data['nitrogen_content'] = np.clip(self.data['nitrogen_content'], 5, 500)
        
        # Phosphorus content (mg/kg)
        self.data['phosphorus_content'] = np.random.lognormal(2.5, 0.7, self.n_samples)
        self.data['phosphorus_content'] = np.clip(self.data['phosphorus_content'], 2, 200)
        
        # Potassium content (mg/kg)
        self.data['potassium_content'] = np.random.lognormal(4, 0.4, self.n_samples)
        self.data['potassium_content'] = np.clip(self.data['potassium_content'], 20, 800)
        
        # Organic matter content (%) - correlated with nitrogen
        base_om = np.random.gamma(2, 1.5, self.n_samples)
        nitrogen_correlation = (self.data['nitrogen_content'] - np.mean(self.data['nitrogen_content'])) / np.std(self.data['nitrogen_content'])
        self.data['organic_matter'] = np.clip(base_om + nitrogen_correlation * 0.5, 0.5, 8.0)
        
        # Electrical conductivity (dS/m) - indicator of soil salinity
        self.data['electrical_conductivity'] = np.random.exponential(0.5, self.n_samples)
        self.data['electrical_conductivity'] = np.clip(self.data['electrical_conductivity'], 0.1, 4.0)
        
    def generate_environmental_factors(self):
        """Generate environmental factors"""
        # Air temperature (°C) - correlated with soil temperature
        temp_correlation = 0.8
        air_temp_base = self.data['soil_temperature'] + np.random.normal(2, 3, self.n_samples)
        self.data['air_temperature'] = air_temp_base
        
        # Relative humidity (%) - inversely correlated with temperature
        base_humidity = 70 - (self.data['air_temperature'] - 15) * 1.5
        self.data['relative_humidity'] = np.clip(base_humidity + np.random.normal(0, 10, self.n_samples), 20, 95)
        
        # Rainfall (mm) - seasonal variation
        seasonal_rainfall = []
        for season in self.data['season']:
            if season in ['Spring', 'Summer']:
                rainfall = np.random.gamma(2, 25)  # More rain in growing season
            else:
                rainfall = np.random.gamma(1.5, 15)  # Less rain in fall/winter
            seasonal_rainfall.append(rainfall)
        self.data['rainfall'] = np.clip(seasonal_rainfall, 0, 200)
        
        # Solar radiation (MJ/m²/day) - seasonal and latitude dependent
        base_radiation = 15 + (45 - self.data['latitude']) * 0.3
        seasonal_radiation_adj = []
        for season in self.data['season']:
            if season == 'Summer':
                adj = np.random.normal(8, 2)
            elif season == 'Winter':
                adj = np.random.normal(-6, 2)
            else:
                adj = np.random.normal(0, 3)
            seasonal_radiation_adj.append(adj)
        
        self.data['solar_radiation'] = np.clip(base_radiation + seasonal_radiation_adj, 5, 35)
        
    def generate_management_factors(self):
        """Generate management and farming practice factors"""
        # Days since last fertilization (0-365)
        self.data['days_since_fertilization'] = np.random.randint(0, 366, self.n_samples)
        
        # Fertilizer type (categorical)
        fertilizer_types = ['NPK', 'Organic', 'Nitrogen', 'Phosphorus', 'None']
        fertilizer_weights = [0.4, 0.2, 0.15, 0.1, 0.15]  # More NPK and organic
        self.data['fertilizer_type'] = np.random.choice(fertilizer_types, self.n_samples, p=fertilizer_weights)
        
        # Irrigation frequency (times per week)
        seasonal_irrigation = []
        for season in self.data['season']:
            if season == 'Summer':
                freq = np.random.poisson(4)  # More irrigation in summer
            elif season in ['Spring', 'Fall']:
                freq = np.random.poisson(2)
            else:  # Winter
                freq = np.random.poisson(1)
            seasonal_irrigation.append(min(freq, 7))  # Max 7 times per week
        self.data['irrigation_frequency'] = seasonal_irrigation
        
        # Crop rotation history (categorical)
        crop_rotations = ['Corn-Soy', 'Wheat-Legume', 'Monoculture', 'Diversified', 'Fallow']
        self.data['crop_rotation'] = np.random.choice(crop_rotations, self.n_samples)
        
    def calculate_soil_health_index(self):
        """Calculate soil health index based on multiple factors"""
        # Normalize key parameters for soil health calculation
        scaler = StandardScaler()
        
        # Key factors for soil health (higher is better for most)
        ph_score = 100 - abs(self.data['soil_ph'] - 6.5) * 20  # Optimal pH around 6.5
        ph_score = np.clip(ph_score, 0, 100)
        
        organic_matter_score = np.clip(self.data['organic_matter'] * 20, 0, 100)  # Higher OM is better
        
        moisture_score = 100 - abs(self.data['soil_moisture'] - 30) * 2  # Optimal around 30%
        moisture_score = np.clip(moisture_score, 0, 100)
        
        nutrient_score = np.clip((
            np.log(self.data['nitrogen_content']) * 10 +
            np.log(self.data['phosphorus_content']) * 8 +
            np.log(self.data['potassium_content']) * 6
        ) / 3, 0, 100)
        
        salinity_score = 100 - self.data['electrical_conductivity'] * 20  # Lower salinity is better
        salinity_score = np.clip(salinity_score, 0, 100)
        
        # Management factor adjustments
        fertilization_bonus = []
        for i, fert_type in enumerate(self.data['fertilizer_type']):
            days = self.data['days_since_fertilization'][i]
            if fert_type == 'None':
                bonus = -5
            elif days < 30:
                bonus = 10 if fert_type == 'Organic' else 5
            elif days < 90:
                bonus = 5 if fert_type == 'Organic' else 2
            else:
                bonus = 0
            fertilization_bonus.append(bonus)
        
        # Calculate weighted soil health index
        soil_health_index = (
            ph_score * 0.20 +
            organic_matter_score * 0.25 +
            moisture_score * 0.20 +
            nutrient_score * 0.25 +
            salinity_score * 0.10 +
            np.array(fertilization_bonus)
        )
        
        # Add some random variation
        soil_health_index += np.random.normal(0, 3, self.n_samples)
        self.data['soil_health_index'] = np.clip(soil_health_index, 0, 100)
        
    def calculate_target_variables(self):
        """Calculate additional target variables"""
        # Crop yield potential (tons/hectare) - based on soil health
        base_yield = self.data['soil_health_index'] / 20  # Scale to 0-5 tons base
        
        # Environmental adjustments
        temp_adjustment = 1 + (25 - abs(self.data['air_temperature'] - 25)) / 50
        rainfall_adjustment = 1 + (self.data['rainfall'] - 50) / 200
        
        self.data['crop_yield_potential'] = np.clip(
            base_yield * temp_adjustment * rainfall_adjustment + np.random.normal(0, 0.5, self.n_samples),
            0.5, 8.0
        )
        
        # Fertilizer recommendation (NPK ratios)
        n_need = np.clip(200 - self.data['nitrogen_content'], 0, 200)
        p_need = np.clip(100 - self.data['phosphorus_content'], 0, 100)
        k_need = np.clip(150 - self.data['potassium_content'], 0, 150)
        
        self.data['fertilizer_n_recommendation'] = n_need
        self.data['fertilizer_p_recommendation'] = p_need
        self.data['fertilizer_k_recommendation'] = k_need
        
        # Irrigation requirement (mm/week)
        base_irrigation = np.clip(50 - self.data['soil_moisture'], 0, 50)
        seasonal_adj = [1.5 if s == 'Summer' else 0.8 if s == 'Winter' else 1.0 for s in self.data['season']]
        self.data['irrigation_requirement'] = base_irrigation * seasonal_adj
        
        # Soil degradation risk (categorical)
        risk_scores = (
            (self.data['soil_health_index'] < 40).astype(int) * 2 +
            (self.data['electrical_conductivity'] > 2.0).astype(int) +
            (self.data['organic_matter'] < 2.0).astype(int) +
            (self.data['bulk_density'] > 1.6).astype(int)
        )
        
        risk_labels = []
        for score in risk_scores:
            if score >= 3:
                risk_labels.append('High')
            elif score >= 1:
                risk_labels.append('Medium')
            else:
                risk_labels.append('Low')
        
        self.data['soil_degradation_risk'] = risk_labels
        
    def generate_complete_dataset(self):
        """Generate the complete dataset"""
        print("Generating location data...")
        self.generate_location_data()
        
        print("Generating temporal data...")
        self.generate_temporal_data()
        
        print("Generating soil physical properties...")
        self.generate_soil_physical_properties()
        
        print("Generating soil chemical properties...")
        self.generate_soil_chemical_properties()
        
        print("Generating environmental factors...")
        self.generate_environmental_factors()
        
        print("Generating management factors...")
        self.generate_management_factors()
        
        print("Calculating soil health index...")
        self.calculate_soil_health_index()
        
        print("Calculating target variables...")
        self.calculate_target_variables()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data)
        
        # Round numerical columns to appropriate decimal places
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in ['soil_health_index', 'crop_yield_potential']:
                df[col] = df[col].round(1)
            elif col in ['soil_ph', 'organic_matter']:
                df[col] = df[col].round(2)
            else:
                df[col] = df[col].round(1)
        
        return df
    
    def save_dataset(self, df, filename='soil_health_dataset.csv'):
        """Save the dataset to CSV file"""
        import os
        
        # Ensure the data directory exists
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        
        # Save summary statistics
        summary_filename = filename.replace('.csv', '_summary.txt')
        summary_filepath = os.path.join(data_dir, summary_filename)
        with open(summary_filepath, 'w') as f:
            f.write("Soil Health Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Features: {len(df.columns)}\n\n")
            f.write("Dataset Info:\n")
            f.write(str(df.info()) + "\n\n")
            f.write("Summary Statistics:\n")
            f.write(str(df.describe()) + "\n\n")
            f.write("Missing Values:\n")
            f.write(str(df.isnull().sum()) + "\n")
        
        print(f"Summary statistics saved to {summary_filepath}")

def main():
    """Main function to generate the dataset"""
    print("Starting soil health dataset generation...")
    
    # Generate dataset
    generator = SoilDataGenerator(n_samples=10000)
    df = generator.generate_complete_dataset()
    
    # Display basic information
    print(f"\nDataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Save dataset
    generator.save_dataset(df)
    
    # Display sample data
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
