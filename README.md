# MIDAS Hackathon - AI for Social Good Projects

This repository contains three AI projects developed for the MIDAS Hackathon, focused on applying machine learning and AI technologies to address real-world challenges in Detroit.

## 🎯 Project Overview

### Project 1: Improving Search and Interaction in Detroit's Open Data Portal
**Directory**: `1_detroit_open_data_portal/`

Natural language chatbot and search enhancement for Detroit's Open Data Portal, enabling intuitive exploration of 200+ city datasets for residents and researchers.

### Project 2: Computer Vision for City Planning in Detroit
**Directory**: `2_detroit_computer_vision/`

Computer vision tools using Detroit imagery (1999–2024) and GIS data to assess building habitability and enhance property-level data accuracy.

### Project 3: Flood and Erosion Risk Policy Analysis Tool
**Directory**: `3_detroit_flood_risk_analysis/`

Interactive policy modeling tool for Detroit's resilience planning, supporting flood and erosion risk analysis under various policy scenarios.

---

## 📊 Project Details

### 💬 Project 1: Detroit Open Data Portal Enhancement
**Challenge**: Navigate 200+ datasets across multiple city departments
**Solution**: Natural language chatbot with synonym recognition and improved metadata mapping
**Impact**: Serves residents seeking localized data and researchers requiring cross-departmental insights

### 🏙️ Project 2: Computer Vision for City Planning  
**Challenge**: Assess building habitability and improve property data accuracy using imagery
**Solution**: Computer vision models to identify uninhabitable structures and assist census audits
**Impact**: Streamlines manual verification and enhances base unit data for multi-family buildings

### 🌊 Project 3: Flood Risk Policy Analysis
**Challenge**: Model flood and erosion risks under various policy scenarios
**Solution**: Interactive policy tool with LLM-based scenario analysis and hydrological modeling
**Impact**: Supports Detroit's resilience planning with Great Lakes-specific data insights

---

## ⚙️ Environment Setup

### Quick Start (Recommended)

1. **Create the conda environment:**
   ```bash
   # Create environment from file
   conda env create -f environment.yml
   
   # Activate environment
   conda activate midas_aug25
   ```

2. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost, jupyter; print('✅ All core libraries installed!')"
   ```

3. **Launch Jupyter for data analysis:**
   ```bash
   jupyter lab
   # Navigate to: 2_detroit_computer_vision/eda/project2_detroit_blight_eda.ipynb
   ```

### Alternative Setup Options

**Using pip only:**
```bash
# Create virtual environment
python -m venv midas_aug25
source midas_aug25/bin/activate  # Windows: midas_aug25\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**What's included in the environment:**
- 🔬 **Data Science**: pandas, numpy, matplotlib, seaborn, jupyter
- 🤖 **Machine Learning**: scikit-learn, xgboost, joblib  
- 🗂️ **Data Handling**: openpyxl (Excel), geopandas (GIS)
- 🤖 **AI/RAG**: langchain, faiss-cpu, openai integrations
- 🗺️ **Geospatial**: folium, contextily (Detroit spatial data)

For detailed setup instructions and troubleshooting, see **[SETUP.md](SETUP.md)**.

## 🚀 Quick Start

Each project directory contains detailed setup and usage instructions. See individual README files for specific implementation details.

```bash
# Explore any project
cd 1_detroit_open_data_portal/     # Open Data Portal project
cd 2_detroit_computer_vision/      # Computer Vision project  
cd 3_detroit_flood_risk_analysis/  # Flood Risk Analysis project
cat README.md
```

## 📁 Repository Structure

```
MIDAS-Hackathon-Aug2025/
├── 1_detroit_open_data_portal/     # Detroit Open Data Portal Enhancement
│   └── README.md
├── 2_detroit_computer_vision/      # Computer Vision for City Planning
│   ├── eda/                       # Exploratory Data Analysis
│   │   └── project2_detroit_blight_eda.ipynb
│   └── README.md
├── 3_detroit_flood_risk_analysis/  # Flood Risk Policy Analysis Tool
│   └── README.md
├── data/                          # Datasets (not committed to git)
│   ├── blight_survey_data/        # DLBA property condition surveys
│   ├── cod_layers_csv/            # City of Detroit address/parcel data
│   └── cod_layers_gdb/            # Spatial geodatabase
├── learning/                      # Learning resources and experiments
│   ├── rag_for_proj1/            # RAG implementation examples
│   └── xgboost_for_proj2/        # XGBoost blight classification
├── environment.yml               # Conda environment specification
├── requirements.txt              # Pip requirements (alternative)
├── SETUP.md                      # Detailed setup instructions
└── README.md                     # This file
```

## 🎯 Social Impact Goals

All three projects demonstrate AI's potential for positive social impact in Detroit:

- **Project 1** enhances civic engagement through improved data accessibility
- **Project 2** supports urban planning and housing safety initiatives  
- **Project 3** advances climate resilience and flood preparedness

These implementations provide practical tools for city officials, residents, and researchers while advancing understanding of applied AI for urban challenges.

## 🤝 Contributing

This hackathon project welcomes contributions to:
- Improve model performance and fairness
- Add real-world data integration
- Enhance educational documentation
- Expand to additional social good applications

## 📜 About

Developed for the MIDAS Hackathon with focus on applying AI and machine learning technologies to address real-world challenges affecting communities and urban environments.

Here's a [link](https://midas.umich.edu/) for general MIDAS resources.

---

*🌟 AI for Social Good - Building technology that serves communities*
