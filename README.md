# Student Predictive Grades

Student Predictive Grades is a Python-based GUI application for training and evaluating machine learning models to predict student exam performance based on behavioural, academic and lifestyle factors.

## Features

The Student Predictive Grades GUI can: 
- Load multiple training datasets (CSV or Excel).
- Clean datasets by:
  - Imputing irregular values with the median value for that column.
  - Dropping other missing values.
  - Capping exam scores at 100.
- Train a `RandomForestRegressor` model with selected features and target.
- Make predictions on a separate test dataset.
- Display the R² score and formatted prediction output.
- Encode known categorical features only (with `LabelEncoder`).

## GUI Overview

| Button                     | Functionality                                               |
|----------------------------|-------------------------------------------------------------|
| `Load Training Dataset(s)` | Load one or more training files into memory                 |
| `Clean Training Dataset(s)`| Clean and prepare training data for modelling               |
| `Load Test Dataset`        | Load a test dataset for prediction                          |
| `Clean Test Dataset`       | Apply cleaning rules to the test dataset                    |
| `Train Model`              | Train model using selected features and target              |
| `Make Predictions`         | Predict exam scores for all students in the test dataset    |

## Machine Learning Details

- Model: `RandomForestRegressor` from `sklearn`
- Evaluation Metric: R² score (coefficient of determination)
- Target variable: `exam_score`
- Categorical features are label-encoded using a fixed list

## Example Features

```text
age, gender, study_hours_per_day, social_media_hours, netflix_hours,
part_time_job, attendance_percentage, sleep_hours, diet_quality,
exercise_frequency, parental_education_level, internet_quality,
mental_health_rating, extracurricular_participation
```

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- openpyxl
- tkinter 

## Getting Started 
1. Clone this repository
2. Run the script: <br>
`python student_predictive_grades.py`
3. Use the GUI to:
    - Load and clean your training files
    - Load and clean the test file
    - Train your model
    - Make predictions

## Current Limitations
- Requires manual cleaning if user skips the "Clean Training Dataset(s)" or "Clean Test Dataset" buttons
- Only supports tabular numeric and categorical data

## Future Improvements
- Enable users to select features with drop down menu/tick boxes
- Add automatic detection of unseen categorical values
- Save/load trained model files