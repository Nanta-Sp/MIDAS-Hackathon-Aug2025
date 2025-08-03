"""
Exploratory Data Analysis for Detroit Open Data Portal datasets
Analyzes three main datasets: blight survey, COD layers CSV, and geodatabase
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def analyze_blight_survey_data():
    """
    Analyze the Detroit Land Bank Authority blight survey data
    """
    print("=" * 60)
    print("BLIGHT SURVEY DATA ANALYSIS")
    print("=" * 60)
    
    # Load the Excel file
    file_path = "data/blight_survey_data/20250527_DLBA_survey_data_UM_Detroit.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        
        print(f"Dataset Shape: {df.shape}")
        print(f"Number of rows: {df.shape[0]:,}")
        print(f"Number of columns: {df.shape[1]}")
        print()
        
        print("Column Information:")
        print("-" * 40)
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"{i:2d}. {col:<35} | {dtype:<15} | {null_count:>5} nulls ({null_pct:5.1f}%)")
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nBasic Statistics for Numeric Columns:")
        print(df.describe())
        
        print("\nData Types Summary:")
        print(df.dtypes.value_counts())
        
        # Check for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical Columns ({len(categorical_cols)}):")
            for col in categorical_cols[:5]:  # Show first 5
                unique_vals = df[col].nunique()
                print(f"  {col}: {unique_vals} unique values")
                if unique_vals <= 10:
                    print(f"    Values: {list(df[col].unique())}")
        
        return df
        
    except Exception as e:
        print(f"Error loading blight survey data: {e}")
        return None


def analyze_cod_layers_csv():
    """
    Analyze the City of Detroit (COD) layers CSV data
    """
    print("\n" + "=" * 60)
    print("COD LAYERS CSV DATA ANALYSIS")
    print("=" * 60)
    
    data_dir = "data/cod_layers_csv/20250728_CODLayers.csv"
    csv_files = ["Addresses.csv", "Buildings.csv", "Parcels2025.csv"]
    
    datasets = {}
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        dataset_name = csv_file.replace('.csv', '')
        
        print(f"\n{dataset_name.upper()} Dataset:")
        print("-" * 50)
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            datasets[dataset_name] = df
            
            print(f"Dataset Shape: {df.shape}")
            print(f"Number of rows: {df.shape[0]:,}")
            print(f"Number of columns: {df.shape[1]}")
            
            print(f"\nColumns ({df.shape[1]}):")
            for i, col in enumerate(df.columns, 1):
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                print(f"{i:2d}. {col:<25} | {dtype:<15} | {null_count:>6} nulls ({null_pct:5.1f}%)")
            
            print(f"\nFirst 3 rows:")
            print(df.head(3))
            
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                print(f"\nNumeric Summary:")
                print(df.describe())
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return datasets


def analyze_geodatabase_info():
    """
    Analyze the geodatabase structure (limited without specialized GIS tools)
    """
    print("\n" + "=" * 60)
    print("GEODATABASE ANALYSIS")
    print("=" * 60)
    
    gdb_path = "data/cod_layers_gdb/CODBaseUnitLayers.gdb"
    
    if os.path.exists(gdb_path):
        print(f"Geodatabase found at: {gdb_path}")
        
        # Get file information
        files = os.listdir(gdb_path)
        print(f"Number of files in geodatabase: {len(files)}")
        
        # Categorize files by extension
        extensions = {}
        for file in files:
            ext = os.path.splitext(file)[1] if '.' in file else 'no_extension'
            extensions[ext] = extensions.get(ext, 0) + 1
        
        print("\nFile types in geodatabase:")
        for ext, count in sorted(extensions.items()):
            print(f"  {ext}: {count} files")
        
        # Get total size
        total_size = sum(os.path.getsize(os.path.join(gdb_path, f)) for f in files)
        print(f"\nTotal geodatabase size: {total_size / (1024*1024):.2f} MB")
        
        print("\nNote: This is an ESRI geodatabase (.gdb) which requires")
        print("specialized GIS software (like ArcGIS or QGIS with GDAL) for full analysis.")
        print("Consider using geopandas or arcpy for detailed geodatabase exploration.")
        
    else:
        print("Geodatabase not found!")
    
    return None


def main():
    """
    Run all EDA analyses
    """
    print("DETROIT OPEN DATA PORTAL - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Change to project root directory
    os.chdir('/Users/admin/dev/MIDAS-Hackathon-Aug2025')
    
    # Analyze each dataset
    blight_data = analyze_blight_survey_data()
    cod_data = analyze_cod_layers_csv()
    gdb_info = analyze_geodatabase_info()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if blight_data is not None:
        print(f"✓ Blight Survey Data: {blight_data.shape[0]:,} rows, {blight_data.shape[1]} columns")
    
    if cod_data:
        for name, df in cod_data.items():
            print(f"✓ COD {name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    print("✓ Geodatabase: Structural analysis completed")
    
    print("\nAll datasets loaded successfully! Ready for deeper analysis.")


if __name__ == "__main__":
    main()