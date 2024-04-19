# IAD Seminar Spring 2024: Day 1 Objectives

## Training and Deployment Objectives

- **Training Models**: Utilize the California housing dataset for model training.
- **Model Deployment**: Implement model deployment using Fast API with Heroku and separately on Streamlit.

## 1. Import Data

1. **File Inspection**:
   - Open the data file manually to review the structure and contents of the columns.

2. **Data Classification**:
   - Identify and categorize the data types within the columns as either numerical or categorical.

## 2. Inspect the Data in Pandas

- Use `df.dtypes` to see if numerical and categorical values have the correct data type.
  - Numerical columns should be float64 or int64
  - Categorical columns should be an object
  - Numerical columns can sometimes be read as an object column this can indicate:
    - incorrect numerical value
    - missing value
    - value could not be coerced
- If there is a mismatch between the expected dtype and Pandas dtype, fix it.
- Use `df.info` to check for null values and note which columns have null values.
  - For null non-target values, you can impute, but it for null target values, you should drop the row.

## 3. Split Data into Train/Test to Prevent Data Leakage

- This step is important so you do not include test data information when training your model.
- Sometimes you have to stratify the data so that the test and train data have the same overall distribution based on a column. This column can already exist or you might have to create it. The reason for doing this is that maybe that column is very important for prediction or classification, so you want the train and test to have the same proportion of values.

## 4. Visualize the Data (if applicable)

- Scatter plot of values?!

## 5. Exploratory Data Analysis

- Use ydata_profiling
- Use `df.corr(numeric_only=True)`

- Inspect all columns:
  - Are all values normally distributed, if not take note
  - What are the min/max values?
  - Is the data capped?
  - Are values scaled in any way?

## 6. Feature Engineering

- Check for features that are correlated to target.
- Check for features that are correlated to each other.
- If the independent variables (features) are too correlated, try to either blend them (PCA, ratio) or drop them.
- Use tools:
  - <https://github.com/WillKoehrsen/feature-selector/>
  - <https://github.com/alteryx/featuretools>

## 7. Preprocessing Data for ML Algorithms