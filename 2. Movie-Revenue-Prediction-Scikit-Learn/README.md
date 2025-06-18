# Predicting Movie Box Office Revenue using Scikit-Learn

This project applies classical machine learning regression models to predict the worldwide gross revenue of movies from the IMDB Top 1000 dataset. The focus is on the complete ML pipeline: feature engineering, model training, comparison, and interpretation using Python and Scikit-learn.

![Image](https://github.com/user-attachments/assets/3c1394c6-90ed-47fc-ba4a-44e6bc12dc43)

## Table of Contents
* [Project Goal](#project-goal)
* [Tech Stack](#tech-stack)
* [Methodology](#methodology)
* [Model Performance](#model-performance)
* [Key Insights](#key-insights)
* [How to Run](#how-to-run)

## Project Goal
The objective is to build and evaluate regression models that can accurately predict a movie's box office earnings (`Gross`) based on its available data, such as genre, runtime, ratings, and the popularity of its cast and director.

## Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn

## Methodology

1.  **Data Preprocessing:**
    *   The project uses a pre-cleaned version of the IMDB dataset from a [previous analysis project](https://github.com/your-username/IMDB-Movie-Analysis-and-Recommender). <!-- Link to your first project! -->
    *   The highly skewed target variable, `Gross`, was log-transformed (`np.log1p`) to normalize its distribution, a crucial step for improving the performance of linear models.

2.  **Feature Engineering:**
    *   **`Total_Popularity`**: A novel feature was engineered to quantify the "star power" of a movie. It's the sum of the log-transformed appearance counts of the director and main stars.
    *   **`Genre_valued`**: A weighted feature was created based on the correlation of genres with box office revenue. High-grossing genres like 'Adventure' and 'Action' were given higher weights.

3.  **Model Training and Comparison:**
    *   Three different regression models were trained and compared:
        *   **Linear Regression** (as a baseline)
        *   **Decision Tree Regressor**
        *   **Random Forest Regressor** (an ensemble model)
    *   The data was split into training (80%) and testing (20%) sets, and features were scaled using `StandardScaler`.
    *   5-fold cross-validation was performed to ensure the models' robustness.

## Model Performance

The Random Forest Regressor was the top-performing model. The predictions were transformed back from log scale to the original dollar scale for metric calculation.

| Model               | RMSE (in millions USD) | MAE      | R² Score |
| ------------------- | ---------------------- | -------- | -------- |
| Linear Regression   | 8.29                   | 3.87     | 0.48     |
| Decision Tree       | 9.21                   | 4.98     | 0.36     |
| **Random Forest**   | **7.43**               | **3.47** | **0.58** |

## Key Insights

- The **Random Forest model** provided the best predictions, indicating that the relationships between features and revenue are complex and non-linear, which ensemble methods are well-suited to capture.
- **Feature importance analysis** from the Random Forest model revealed that:
  - `No_of_Votes` is the single most powerful predictor of gross revenue.
  - The engineered `Total_Popularity` feature was the second most important, confirming that star power is a significant driver of financial success.
  - The `Genre_valued` feature also ranked highly, showing that certain genres are inherently more profitable.
