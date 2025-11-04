# lab2_soil_analysis.py

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the soil test dataset from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found. Ensure the file exists at the specified path: {file_path}")
        return None

def clean_data(df):
    """
    Clean the dataset by handling missing values and removing outliers from 'soil_ph'.
    
    For each column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
    - Missing values are filled with the column mean.
    
    Additionally, remove outliers in 'soil_ph' that are more than 3 standard deviations from the mean.
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()
    
    # Fill missing values in each specified column with the column mean
    for col in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
        if df_cleaned[col].isnull().any():
            mean_val = df_cleaned[col].mean()
            df_cleaned[col].fillna(mean_val, inplace=True)
            print(f"Filled missing values in '{col}' with mean value {mean_val:.2f}")
    
    # Remove outliers in 'soil_ph': values more than 3 standard deviations from the mean
    ph_mean = df_cleaned['soil_ph'].mean()
    ph_std = df_cleaned['soil_ph'].std()
    lower_bound = ph_mean - 3 * ph_std
    upper_bound = ph_mean + 3 * ph_std
    df_cleaned = df_cleaned[(df_cleaned['soil_ph'] >= lower_bound) & (df_cleaned['soil_ph'] <= upper_bound)]
    
    print(f"After cleaning, 'soil_ph' values are within the range [{lower_bound:.2f}, {upper_bound:.2f}].")
    print(df_cleaned.head())
    return df_cleaned

def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute statistics.
    """
    min_val = df[column].min()
    max_val = df[column].max()
    mean_val = df[column].mean()
    median_val = df[column].median()
    std_val = df[column].std()
    
    print(f"\nDescriptive statistics for '{column}':")
    print(f"  Minimum: {min_val}")
    print(f"  Maximum: {max_val}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")

def main():
    # Update the file path as needed (relative to your script's location)
    file_path = '../../datasets/soil_test.csv'
    
    # Load the dataset
    df = load_data(file_path)
    if df is None:
        return
    
    # Clean the dataset
    df_clean = clean_data(df)
    
    # Compute and display statistics for the 'soil_ph' column
    compute_statistics(df_clean, 'soil_ph')
    
if __name__ == '__main__':
    main()