# Movie Analytics: Clustering & Classification with Scikit-Learn

This project explores the IMDB Top 1000 movies dataset using both unsupervised and supervised machine learning techniques. The goal is to first uncover natural groupings of movies using K-Means clustering and then to build a classification model to predict whether a movie is "highly-rated."

## Table of Contents
* [Project Overview](#project-overview)
* [Tech Stack](#tech-stack)
* [Part 1: Unsupervised Clustering (K-Means)](#part-1-unsupervised-clustering-k-means)
* [Part 2: Supervised Classification (Logistic Regression)](#part-2-supervised-classification-logistic-regression)
* [Key Results & Insights](#key-results--insights)
* [How to Run](#how-to-run)

## Project Overview

The project is structured in two main parts:
1.  **Clustering:** K-Means is applied to segment movies based on financial (`Log_Gross`, `Runtime`) and quality (`IMDB_Rating`, `Meta_score`) metrics. The optimal number of clusters was determined using the Elbow Method and Silhouette Score.
2.  **Classification:** A Logistic Regression model is trained to predict a binary target: `Is_Highly_Rated` (1 if IMDB Rating ≥ 8.0, else 0). The model's performance is evaluated, and feature importance is analyzed through its coefficients.

## Tech Stack
- Python
- Scikit-learn (KMeans, LogisticRegression, RandomForestClassifier, DBSCAN)
- Pandas & NumPy
- Matplotlib & Seaborn

## Part 1: Unsupervised Clustering (K-Means)
- **Goal:** Identify distinct movie archetypes without pre-existing labels.
- **Method:**
  - Features (`IMDB_Rating`, `Meta_score`) were scaled using `StandardScaler`.
  - The optimal number of clusters (k=4) was validated using the Elbow Method.
- **Result:** Four distinct "quality" clusters were identified:
  - **Cluster 0:** Average Performers
  - **Cluster 1:** Solid Hits
  - **Cluster 2:** Audience Favorites
  - **Cluster 3:** Critically Acclaimed Masterpieces

## Part 2: Supervised Classification (Logistic Regression)
- **Goal:** Predict if a movie will be highly rated by the audience.
- **Method:**
  - A binary target `Is_Highly_Rated` was created.
  - Features included `Meta_score`, `No_of_Votes`, `Released_Year`, and binarized genres.
- **Result:** The model achieved an accuracy of **70.5%** on the test set. *(Replace with your accuracy)*

## Key Results & Insights
- **Clustering Insights:** The analysis successfully segmented movies into meaningful groups, which could be used for targeted marketing or content acquisition strategies. For instance, "Audience Favorites" have high user ratings but not necessarily the highest critic scores.
- **Classification Insights:** The logistic regression model's coefficients confirmed that a high `Meta_score` (critic rating) and a large `No_of_Votes` (popularity) are the strongest predictors of a high IMDB rating.
