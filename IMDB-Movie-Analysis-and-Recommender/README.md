# IMDB Top 1000: Data Analysis and Content-Based Recommender

 https://imgur.com/a/w7ddoEM

## Table of Contents
* [Project Overview](#project-overview)
* [Tech Stack](#tech-stack)
* [Key Features](#key-features)
* [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Content-Based Recommendation System](#content-based-recommendation-system)
* [How to Run](#how-to-run)
* [Key Learnings & Takeaways](#key-learnings--takeaways)

## Project Overview

This project performs a comprehensive analysis of the IMDB Top 1000 movies dataset. The primary goals were to practice a full data science workflow, including data cleaning, feature engineering, exploratory data analysis, and building a foundational recommendation system. The analysis uncovers trends in top-rated films and culminates in a content-based movie recommender that suggests similar movies based on a weighted combination of numerical and categorical features.

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Key Features

### Data Cleaning & Preprocessing
- **Handling Missing Values:** Compared multiple strategies (row deletion vs. imputation) and implemented advanced imputation for key features. `Certificate` was imputed based on genre medians, and `Gross` revenue was imputed based on the mean of movies with a similar number of votes.
- **Data Transformation:** Standardized inconsistent data formats, such as converting `Runtime` and `Gross` columns to numeric types.
- **Feature Engineering:** Created a numerical mapping for the `Certificate` column (e.g., 'PG-13' -> 13) to allow for quantitative analysis. Converted the `Genre` string into a list of genres for multi-label analysis.

### Exploratory Data Analysis (EDA)
- Extracted and visualized over 10 key insights from the data, including:
  - The positive but imperfect correlation between critic scores (`Metascore`) and user scores (`IMDB_Rating`).
  - The distribution of movie runtimes, showing most top films are between 100-140 minutes.
  - Identification of the most prolific genres, with 'Drama' being the most common.
  - Analysis of top director-actor collaborations that consistently produce highly-rated films.

### Content-Based Recommendation System
- Developed a non-ML recommendation engine from scratch.
- The system calculates a custom similarity score between movies using a weighted average of:
  1.  **Numerical Feature Similarity:** Cosine similarity on a normalized vector of features (`Runtime`, `IMDB_Rating`, `Meta_score`, etc.).
  2.  **Genre Similarity:** Jaccard similarity on the set of genres.
  3.  **Personnel Similarity:** Jaccard similarity on the director and main stars.

## Key Learnings & Takeaways
- Gained practical experience in handling real-world, messy data.
- Learned the importance of feature-context in choosing an imputation strategy over simple statistical measures.
- Solidified skills in data visualization with Matplotlib and Seaborn to effectively communicate findings.
- Understood the logic behind content-based filtering by manually constructing the similarity metrics.