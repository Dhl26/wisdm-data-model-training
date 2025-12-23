# wisdm-data-model-training


## Overview
This project implements a machine learning pipeline to classify human activities (Walking, Jogging, Sitting, Standing, Upstairs, Downstairs) using tri-axial accelerometer data from the WISDM (Wireless Sensor Data Mining) dataset.


## Setup & Installation
1.  **Prerequisites**: Python 3.x
2.  **Dependencies**: Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib scikit-learn python-docx
    ```

## Usage Instructions

### 1. Data Preprocessing
Run the preprocessing script to clean the raw data, perform EDA, and extract features.
```bash
python "eda_preprocessing.py"
```
*   **Input**: `WISDM_ar_v1.1_raw.txt`
*   **Output**: 
    *   `processed_wisdm_data.csv` (Features)
    *   `activity_distribution.png` (Plot)
    *   `time_series_example.png` (Plot)
*   **Key Steps**:
    *   Cleans formatting artifacts (trailing semi-colons).
    *   Removes duplicates and fills missing values.
    *   Segments data into 10-second sliding windows (200 samples).
    *   Extracts statistical features: Mean, Standard Deviation, Max, Min, MAD.

### 2. Model Training & Evaluation
Run the modeling script to train Random Forest and Gradient Boosting classifiers.
```bash
python "modeling.py"
```
*   **Input**: `processed_wisdm_data.csv`
*   **Output**:
    *   Console output with Accuracy, Precision, Recall, and F1-Scores.
    *   `confusion_matrix_Random_Forest.png`
    *   `feature_importance.png`
*   **Key Steps**:
    *   Splits data into Train/Test sets (Stratified).
    *   Scales features using StandardScaler.
    *   Trains Random Forest and Gradient Boosting models.
    *   Evaluates performance and visualizes results.


## Results
*   **Best Model**: Random Forest (~69% Accuracy)
*   **Key Insight**: The **Standard Deviation** of the accelerometer signal is the most critical feature for distinguishing between static (Sitting) and dynamic (Jogging) activities.

## Dataset Information
*   **Source**: WISDM Lab, Fordham University.
*   **Sampling Rate**: 20Hz.
*   **Subjects**: 36 users.
*   **Activities**: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing.
