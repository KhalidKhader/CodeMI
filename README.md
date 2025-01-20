# Data Analysis and Metric Evaluation Project

This repository contains the complete workflow for analyzing data, implementing custom metrics, and visualizing insights. The project focuses on exploring datasets through various statistical and machine learning techniques, including clustering, metric evaluations, and data distribution analyses.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Figures and Visualizations](#figures-and-visualizations)
- [Installation and Usage](#installation-and-usage)
- [Insights and Conclusions](#insights-and-conclusions)
- [Acknowledgments](#acknowledgments)

---

## Introduction

This project involves:
- Clustering data into distinct groups.
- Evaluating custom metrics for data analysis.
- Visualizing distributions, correlations, and other statistical insights.
- Highlighting the effectiveness of new and complex metrics.

The work is designed to provide a detailed and structured approach to analyzing datasets and extracting actionable insights.

---

## Features

1. **Clustering Analysis**:
   - Clustering data into 3 distinct groups based on mutual information (MI).

2. **Metric Implementation**:
   - Development and evaluation of complex metrics to measure dataset properties.

3. **Data Visualization**:
   - Histogram and boxplot visualizations for feature distribution and variability.
   - Scatter plots, heatmaps, and bar plots for feature relationships and comparisons.

4. **Conclusions**:
   - Insights based on visualizations and analysis, emphasizing key metrics.

---

## Project Structure

The repository is organized as follows:

```
.
├── Figures/
│   └── Analysis figures/
│       ├── clustring3clustres_vs_mi.png
│       ├── complex_metrix_1.png
│       ├── complex_metrix_2.png
│       ├── new_metrix.png
│       ├── dist_AMI.png
│       ├── max_ind.png
│       ├── [Additional generated figures...]
├── report/
│   └── main.tex  # Main LaTeX file for the report
├── README.md
├── requirements.txt  # Python dependencies
├── analysis.ipynb  # Jupyter Notebook for data analysis
└── [Other supporting files...]
```

---

## Figures and Visualizations

### Key Figures
1. **Clustering**:
   - `clustring3clustres_vs_mi.png`: Highlights 3 clusters based on MI.

2. **Metrics**:
   - `complex_metrix_1.png` and `complex_metrix_2.png`: Complex metrics visualizations.
   - `new_metrix.png`: The best-performing metric visualization.

3. **Data Distribution**:
   - `dist_AMI.png`: Distribution of the last metric.
   - `max_ind.png`: Histogram of max indentation.

---

## Installation and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Dependencies**:
   Install the required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Analysis**:
   Open the Jupyter Notebook:
   ```bash
   jupyter notebook analysis.ipynb
   ```

4. **Generate the Report**:
   Compile the LaTeX report:
   ```bash
   cd report
   pdflatex main.tex
   ```

---

## Insights and Conclusions

- **Clustering Analysis**:
  MI successfully divides the data into 3 distinct clusters, offering insights into data separability.

- **Metric Evaluations**:
  - Complex metrics (`complex_metrix_1.png` and `complex_metrix_2.png`) provided valuable insights, but the new metric (`new_metrix.png`) outperformed others in clarity and interpretability.

- **Data Distribution**:
  - The distribution of the last metric (`dist_AMI.png`) and max indentation histogram (`max_ind.png`) emphasize the variability and structure of the dataset.

These findings highlight the effectiveness of the applied methodologies in extracting meaningful insights.

---

## Acknowledgments

This project reflects the culmination of detailed data analysis and visualization efforts. A special thanks to [Abdullah Afifi/Software construction course with Dr. Majed Ayad Fall 2025 - Master -BZU students] for the support and resources.
