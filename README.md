# 🎬 Netflix Director Style Analysis

## ✅ Function Definition

This project analyzes the styles of directors based on Netflix content metadata and builds a genre classification model to predict the genre of new content.

- Preprocessing Netflix metadata and generating unified genre labels
- KMeans clustering based on average content features by director
- Genre classification using Random Forest / Decision Tree models
- Hyperparameter tuning using GridSearchCV
- Model evaluation and visualization (Confusion Matrix, F1-score, etc.)
- Cross-validation based performance analysis

## ✅ Architecture

```
📁 netflix_director_style/
├── preprocessing.py             # Preprocess raw CSV into final training-ready format
├── Clustering_KMeans.py         # KMeans clustering based on average director content features
├── Classification_DecisionTree.py  # Genre classification using Decision Tree
├── Random_Forest.py             # Genre classification using Random Forest
├── GridSearch_Tuning.py         # Hyperparameter tuning for Random Forest
├── cross_validation.py          # Cross-validation using Decision Tree (F1-macro)
├── Evaluation_Metrics.py        # Evaluation functions and visualization
├── netflix_titles.csv           # Raw input dataset
├── netflix_preprocessed_final.csv  # Final preprocessed dataset
```

## 📊 Data Flow Structure

```
[netflix_titles.csv]
        ↓ (preprocessing.py)
[netflix_preprocessed_final.csv]
        ↓
    ┌──────────────────────────────┐
    │      Clustering_KMeans.py    │
    └──────────────────────────────┘
                  ↓
    ┌──────────────────────────────┐
    │   Classification Models      │
    │   - DecisionTree             │
    │   - RandomForest             │
    └──────────────────────────────┘
                  ↓
    ┌──────────────────────────────┐
    │   Evaluation_Metrics.py      │
    │   GridSearch_Tuning.py       │
    │   cross_validation.py        │
    └──────────────────────────────┘
```

## ✅ Execution-Based Architecture

```
[netflix_titles.csv]
        ↓ (1. Preprocessing.py)
[netflix_preprocessed_final.csv]
        ↓
    ┌────────────────────────────────────┐
    │     2-1. GridSearch_Tuning.py      │  ← Hyperparameter tuning for Decision Tree
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │   2-2. Classification_DecisionTree │  ← Train and evaluate Decision Tree model
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │     2-3. Random_Forest.py          │  ← Train and evaluate Random Forest model
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │     3. cross_validation.py         │  ← Evaluate model with cross-validation
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │     4. Clustering_KMeans.py        │  ← Cluster directors using KMeans
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │     5. Evaluation_Metrics.py       │  ← Classification & clustering evaluation
    └────────────────────────────────────┘
```

## ✅ Execution Order Summary

| Step | Script                        | Purpose                                              |
|------|-------------------------------|------------------------------------------------------|
| 1    | Preprocessing.py              | Preprocess and save the original dataset             |
| 2-1  | GridSearch_Tuning.py          | Hyperparameter tuning for Decision Tree              |
| 2-2  | Classification_DecisionTree.py| Train and evaluate Decision Tree model               |
| 2-3  | Random_Forest.py              | Train and evaluate final Random Forest model         |
| 3    | cross_validation.py           | Evaluate model generalization via cross-validation   |
| 4    | Clustering_KMeans.py          | Visual clustering of director styles via KMeans      |
| 5    | Evaluation_Metrics.py         | Functions to evaluate classification and clustering  |


## ✅ EDA & Visualization

```
📁 EDA_python_code/
├── FirstStep_boxplot_histogram.py     # Visualizes the distribution of key variables using boxplots and histograms
├── secondStep_heatmap.py              # Creates a heatmap to show correlations between key variables
├── thirdStep_boxplot.py               # Compares variable differences across clusters using boxplots
├── fifthStep_confusion_matrix.py      # Visualizes the confusion matrix of the classification model
├── sixthStep_heatmap.py               # Displays a heatmap of correlations between directors and genres
```


This project includes multiple visualization steps to better understand director styles and evaluate model performance using Netflix content data.

• Visualizing the distribution of key variables using boxplots and histograms
• Correlation heatmap between key variables to examine feature relationships
• Boxplot comparison across director clusters to interpret stylistic differences
• Confusion matrix visualization for evaluating genre prediction accuracy
• Heatmap of director-genre relationships for identifying dominant genre patterns


