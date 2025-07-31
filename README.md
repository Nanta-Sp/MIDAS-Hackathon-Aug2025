# MIDAS Hackathon - AI for Social Good Projects

This repository contains two distinct AI projects developed for the MIDAS Hackathon, focused on applying machine learning and AI technologies to address real-world social challenges.

## 🎯 Project Overview

### Project 1: RAG Systems for Enhanced Information Access
**Directory**: `proj1_chatbots_for_mich/`

Educational implementations of Retrieval-Augmented Generation (RAG) systems, demonstrating both manual and framework-based approaches for building intelligent question-answering systems.

### Project 2: Blight Detection for Urban Safety
**Directory**: `proj2_blight_classification/` 

Machine learning system for detecting and classifying property blight severity to support municipal intervention efforts and community safety initiatives.

---

## 💬 Project 1: RAG Systems Comparison

### Problem Statement
Information access and knowledge management are critical challenges in many domains. RAG systems enhance AI responses by combining retrieval of relevant information with natural language generation.

### Our Solution
Two complete RAG implementations for educational and production use:

1. **Manual RAG** (`medium_manual_rag/`): Built from scratch for learning
   - Custom TF-IDF embeddings and retrieval
   - Template-based generation
   - Zero external dependencies
   - ~800 lines of educational code

2. **LangChain RAG** (`easy_langchain_rag/`): Modern framework implementation
   - OpenAI embeddings and GPT integration
   - FAISS vector database
   - Production-ready architecture
   - ~30 lines of framework code

### Educational Value
Provides clear learning progression from fundamental algorithms to modern AI frameworks, with comprehensive documentation and examples for computer science education.

---

## 🏠 Project 2: Blight Detection System

### Problem Statement
Property blight poses serious threats to public health, safety, and neighborhood stability. In Detroit alone, blighted properties contribute to:
- Reduced neighborhood quality of life and community pride
- Public safety hazards and health risks
- Economic decline and reduced property values
- Community disinvestment and population loss

### Our Solution
A comprehensive XGBoost-based classification system that predicts property blight severity on a 4-level scale:

- **Level 1**: No blight (well-maintained property)
- **Level 2**: Minor blight (cosmetic issues, minor repairs needed)
- **Level 3**: Moderate blight (structural issues, significant deterioration)  
- **Level 4**: Severe blight (unsafe/uninhabitable, potential demolition)

### Impact & Applications
This system helps municipalities:
- **Prioritize inspections** based on blight risk scores
- **Allocate resources** efficiently to highest-risk properties
- **Support community planning** with data-driven insights
- **Track progress** of neighborhood improvement initiatives

The model achieves 59.5% accuracy with excellent screening capabilities (97.5% recall for identifying non-blighted properties), making it valuable for municipal resource allocation.

### Technical Approach
- **Data Sources**: 22 features from property records, tax data, code enforcement, USPS vacancy indicators, and neighborhood demographics
- **Model**: XGBoost multi-class classifier with comprehensive evaluation metrics  
- **Scale**: Trained on datasets up to 100,000 properties
- **Deployment**: Production-ready prediction pipeline with batch processing capabilities

---

## 🚀 Quick Start

### Project 1: RAG Systems
```bash
cd proj1_chatbots_for_mich

# Educational path - start with manual implementation
cd medium_manual_rag
python simple_rag.py

# Production path - modern framework
cd ../easy_langchain_rag
pip install -r requirements.txt
python langchain_rag.py
```

### Project 2: Blight Detection
```bash
cd proj2_blight_classification
pip install -r requirements.txt

# Generate synthetic data
python generate_synthetic_data.py

# Train models
python train_blight_model.py

# Make predictions
python predict_blight.py --model models/blight_model_large_dataset.joblib --demo
```

## 📁 Repository Structure

```
MIDAS-Hackathon-Aug2025/
├── proj1_chatbots_for_mich/       # RAG Systems Project
│   ├── medium_manual_rag/         # Educational RAG implementation
│   ├── easy_langchain_rag/        # Production RAG with LangChain
│   └── README.md
├── proj2_blight_classification/   # Blight Detection Project  
│   ├── data/                      # Generated datasets
│   ├── models/                    # Trained XGBoost models
│   ├── generate_synthetic_data.py # Dataset creation
│   ├── train_blight_model.py     # Model training pipeline
│   ├── predict_blight.py         # Prediction interface
│   └── README.md
└── README.md                      # This file
```

## 🎯 Social Impact Goals

Both projects demonstrate AI's potential for positive social impact:

- **Project 1** enhances information accessibility and AI education
- **Project 2** directly supports urban safety and community improvement efforts

These implementations provide practical tools while advancing understanding of applied machine learning for social good.

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
