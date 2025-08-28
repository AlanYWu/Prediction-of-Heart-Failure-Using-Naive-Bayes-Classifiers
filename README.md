# Prediction of Heart Failure Using Naive Bayes Classifiers

A machine learning project that implements a **Naive Bayes Classifier from scratch** to predict heart failure based on medical data.

## 🎯 Project Overview

This project demonstrates the mathematical foundations of Naive Bayes classification by building the algorithm entirely from scratch, without relying on pre-built machine learning libraries for the core classification logic.

## 🔬 What I Built

- **Custom Naive Bayes Implementation**: Developed the complete Naive Bayes algorithm from mathematical principles
- **Dual Data Type Support**: Handles both categorical and numerical features simultaneously
- **Feature Engineering**: Implemented separate probability calculations for different data types
- **Performance Metrics**: Achieved **88.4% accuracy** on heart disease prediction

## 📊 Feature Overview

![Feature Overview](Graph/Data%20representation/Picture1.png)

## 🛠️ Technical Implementation

The classifier processes:
- **Numerical features**: Uses normal distribution assumptions with mean and standard deviation
- **Categorical features**: Calculates conditional probabilities for each category
- **Combined prediction**: Merges probabilities from both feature types for final classification

## 📊 Results

- **Accuracy**: 88.4%
- **Algorithm**: Naive Bayes Classifier (implemented from scratch)
- **Dataset**: Heart disease prediction with mixed categorical and numerical features

## 📁 Project Structure

- `Naive Bayes Classifier.py` - Main implementation
- `Data/` - Heart disease datasets
- `Feature processing.ipynb` - Data preprocessing
- `Graph/Data representation/` - Visualization notebooks

This project serves as both a practical heart disease prediction tool and an educational resource for understanding the mathematical foundations of Naive Bayes classification.
