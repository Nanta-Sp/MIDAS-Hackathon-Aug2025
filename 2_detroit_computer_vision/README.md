# Project 2: Computer Vision for City Planning in Detroit

## Overview

This project leverages computer vision and machine learning to analyze Detroit's urban landscape using historical and contemporary imagery (1999–2024) combined with GIS data. The goal is to build tools that enhance city planning capabilities and improve the accuracy of property-level data throughout Detroit.

## Problem Statement

Detroit faces significant challenges in maintaining accurate property assessments and urban planning data:
- **Manual verification processes** are time-intensive and resource-heavy
- **Outdated building habitability assessments** compromise public safety
- **Inaccurate housing unit counts** affect census data and resource allocation
- **Limited automation** in property condition monitoring

## Goals

Create computer vision models and tools to:

### Building Habitability Assessment
- **Identify uninhabitable structures** from aerial and street-level imagery
- **Classify building conditions** across safety and habitability metrics
- **Prioritize inspection resources** based on automated risk assessment
- **Track changes over time** using multi-temporal imagery analysis

### Property Data Enhancement
- **Improve base unit accuracy** for multi-family buildings
- **Assist with census audits** by validating housing unit counts
- **Automate property characteristic detection** (building materials, roof condition, lot usage)
- **Support zoning and land use planning** with updated property classifications

### Predictive Modeling and Automation
- **Streamline manual verification** through automated pre-screening
- **Reduce inspection costs** by targeting high-priority properties
- **Provide early warning systems** for deteriorating building conditions
- **Enable data-driven policy decisions** with accurate property intelligence

## Technical Approach

### Data Sources
- **Historical imagery**: Detroit aerial and satellite imagery (1999-2024)
- **Street-level imagery**: Google Street View, municipal inspection photos
- **GIS data**: Property boundaries, zoning information, building footprints
- **Administrative records**: Permit data, violation history, ownership records

### Computer Vision Models
- **Object detection**: Identify buildings, structures, and property features
- **Classification models**: Assess building condition and habitability status  
- **Segmentation**: Precise building boundary detection and land use classification
- **Change detection**: Temporal analysis to track property condition evolution

### Machine Learning Pipeline
- **Feature extraction**: Automated identification of relevant building characteristics
- **Multi-modal fusion**: Combine imagery with GIS and administrative data
- **Model validation**: Ground-truth comparison with inspection records
- **Scalable deployment**: Production-ready inference for city-wide analysis

## Expected Impact

### For City Officials
- **Enhanced inspection efficiency**: Target resources to highest-risk properties
- **Improved data accuracy**: Better foundation for policy and planning decisions
- **Cost reduction**: Automated pre-screening reduces manual workload
- **Public safety**: Earlier identification of dangerous structures

### For Urban Planning
- **Accurate housing inventory**: Reliable data for demographic and economic analysis
- **Development insights**: Understanding of neighborhood change patterns
- **Resource allocation**: Data-driven distribution of city services and investments
- **Policy evaluation**: Quantitative assessment of intervention effectiveness

### For Research and Community
- **Academic research**: High-quality dataset for urban studies
- **Community engagement**: Transparent, data-driven neighborhood assessments
- **Grant applications**: Evidence base for funding proposals
- **Historical analysis**: Long-term trends in urban development

## Technical Deliverables

1. **Computer vision models** for building condition assessment
2. **Automated processing pipeline** for imagery analysis
3. **Integration tools** for GIS and administrative data
4. **Web interface** for city staff to access results
5. **API endpoints** for integration with existing city systems
6. **Documentation and training** materials for city personnel

## Success Metrics

- **Accuracy**: >90% agreement with manual inspections on habitability assessments
- **Coverage**: City-wide analysis capability for all Detroit properties
- **Efficiency**: 10x reduction in time required for property condition surveys
- **Cost savings**: Measurable reduction in inspection and verification costs
- **Adoption**: Integration into existing city planning workflows