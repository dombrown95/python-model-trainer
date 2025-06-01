import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_FEATURES = [
    'gender',
    'part_time_job',
    'diet_quality',
    'parental_education_level',
    'internet_quality',
    'extracurricular_participation'
]

# Function to initialise and manage global variables
def initialise_globals():
    global df, model, df_predict
    df = None
    model = None
    df_predict = None

# Reads .csv or excel file(s) and loads them into a pandas dataframe and returns an appropriate message.
def load_dataset():
    global df
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    if file_paths:
        dfs = []
        try:
            for path in file_paths:
                if path.endswith('.csv'):
                    dfs.append(pd.read_csv(path))
                else:
                    dfs.append(pd.read_excel(path, engine='openpyxl'))
            df = pd.concat(dfs, ignore_index=True)
            messagebox.showinfo("Success", "Training datasets loaded and combined.")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    return None

# Function to clean the loaded training dataset(s)
def clean_dataset():
    global df
    if df is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return

    original_rows = len(df)

    # Drops rows where age is set to 'unknown'
    if 'age' in df.columns:
        df = df[df['age'] != 'unknown']

    # Drop rows where 'study_hours_per_day' is 'varies'
    if 'study_hours_per_day' in df.columns:
        df = df[df['study_hours_per_day'] != 'varies']

    # Drops rows with other missing values.
    df.dropna(inplace=True)

    # Caps exam_score values at 100.
    if 'exam_score' in df.columns:
        df.loc[df['exam_score'] > 100, 'exam_score'] = 100

    cleaned_rows = len(df)
    removed = original_rows - cleaned_rows
    messagebox.showinfo("Cleaned", f"Training dataset(s) cleaned.\nRows removed: {removed}")

# Function that reads the target .csv or excel file and loads it into the pandas dataframe
def load_prediction_dataset():
    global df_predict
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df_predict = pd.read_csv(file_path)
            else:
                df_predict = pd.read_excel(file_path, engine='openpyxl')
            messagebox.showinfo("Success", "Prediction dataset loaded successfully.")
            return df_predict
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load prediction dataset: {e}")
    return None

# Function to clean the loaded test dataset
def clean_test_dataset():
    global df_predict
    if df_predict is None:
        messagebox.showerror("Error", "Please load a test dataset first.")
        return

    original_rows = len(df_predict)

    # Drops rows where age is set to 'unknown'
    if 'age' in df_predict.columns:
        df_predict = df_predict[df_predict['age'] != 'unknown']

    # Drop rows where 'study_hours_per_day' is 'varies'
    if 'study_hours_per_day' in df_predict.columns:
        df_predict = df_predict[df_predict['study_hours_per_day'] != 'varies']

    # Drop rows with other missing values
    df_predict.dropna(inplace=True)

    cleaned_rows = len(df_predict)
    removed = original_rows - cleaned_rows

    messagebox.showinfo("Cleaned", f"Test dataset cleaned.\nRows removed: {removed}")

# Function that trains a random forest model
def train_model(df, features, target):
    global model
    try:
        if df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return None
        if not features or not target:
            messagebox.showerror("Error", "Please specify both features and target.")
            return None
        
        # Drop rows where target or any feature contains missing values
        X = df[features]
        y = df[target]
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[features]
        y = combined[target]

        # Encodes categorical features (e.g. gender)
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        messagebox.showinfo("Model Trained", f"Model trained successfully! R² Score: {r2:.2f}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
    return None

# Function that uses the trained model to make predictions on new input data and displays the results in the GUI.
def make_predictions(model, df_predict, features):
    try:
        X_new = df_predict[features].copy()

        # Drops rows with missing values in the test dataset.
        X_new.dropna(inplace=True) 

        # Encodes categorical features.
        for col in CATEGORICAL_FEATURES:
            if col in X_new.columns:
                le = LabelEncoder()
                X_new[col] = le.fit_transform(X_new[col].astype(str))

        predictions = model.predict(X_new)
        result_text.delete(1.0, tk.END)
        if 'student_id' not in df_predict.columns:
            messagebox.showerror("Error", "'student_id' column not found in test dataset.")
            return

        student_ids = df_predict['student_id'].tolist()

        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Predicted Grades by Student ID:\n\n")

        for sid, pred in zip(student_ids, predictions):
            result_text.insert(tk.END, f"Student ID {sid}: Predicted Grade = {pred:.2f}\n")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

initialise_globals()

# Initialises the main tktinter window and sets its title.
root = tk.Tk()
root.title("Student Predictive Grades")

# Creates a button that allows the user to load training dataset files.
load_button = tk.Button(root, text="Load Training Dataset(s)", command=lambda: load_dataset())
load_button.pack(pady=10)

# Creates a button that allows the user to clean training dataset files.
clean_button = tk.Button(root, text="Clean Training Dataset(s)", command=clean_dataset)
clean_button.pack(pady=5)

# Creates a button that allows the user to load a test dataset file.
predict_data_button = tk.Button(root, text="Load Test Dataset", command=load_prediction_dataset)
predict_data_button.pack(pady=5)

# Creates a button that allows the user to clean a test dataset file.
clean_test_button = tk.Button(root, text="Clean Test Dataset", command=clean_test_dataset)
clean_test_button.pack(pady=5)

# Creates a label and input field where the user specifies the feature column names (seperated by commas).
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# Creates a label and input field where the user specifies the target column name.
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Creates a button to trigger model training using the selected dataset, features and target.
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(df, [col.strip() for col in features_entry.get().split(',')], target_entry.get()))
train_button.pack(pady=10)

# Creates a button to trigger prediction using the trained model and selected features.
predict_button = tk.Button(root, text="Make Predictions", command=lambda: make_predictions(model, df_predict, [col.strip() for col in features_entry.get().split(',')]))
predict_button.pack(pady=10)

# Creates a text box to display model prediction results in the GUI.
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Starts the tkinter main event loop to display the GUI.
root.mainloop()