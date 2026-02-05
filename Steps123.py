# %% [markdown]
# # Machine Learning Analytics

# %% [markdown]
# ## Step 1: Define the Problem
# Review these two datasets and brainstorm problems that could be addressed with the dataset. Identify a question for each dataset.
# - College Completion, Is there a correlation between the institutional level of schooling and the graduation rate?
# -  Job Placement, Does gender have a significant impact on job salary, despite having the same qualifications?

# %% [markdown]
# ## Step 2: Work through the steps outlined in the examples to include the following elements:
# - Write a generic question that this dataset could address.
# - What is a independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# - Data preparation:
#  -- correct variable type/class as needed
#  -- collapse factor levels as needed
#  -- one-hot encoding factor variables
#  -- normalize the continuous variables
#  -- drop unneeded variables
#  -- create target variable if needed
#  -- Calculate the prevalence of the target variable
#  -- Create the necessary data partitions (Train,Tune,Test)

# %% [markdown]
# ## Generic Questions
# - College Completion, Is there a correlation between the institutional level of schooling and the graduation rate?
# - Job Placement, Does gender have a significant impact on job salary, despite having the same qualifications?

# %% [markdown]
# ## Independent Business Metric
# ### The measure we use to track whether the algorithm we have built is delivering value for our organization.
# - College Completion: The independent business metric is the graduation rate (the percentage of students who complete their college education). This dependent variable helps us understand institutional success rates and will be analyzed against the level (whether that be 2-year vs 4-year) to find correlations between institution type and the graduation rates.
# - Job Placement: The independent business metric is average salary of job placements. This dependent variable helps us evaluate financial outcomes and will be analyzed by based on gender (while keeping qualifications controlled) to assess whether gender disparities exist in salary outcomes for similarly qualified people.

# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %%
## Correct variable type/class as needed
# College data
college_data = pd.read_csv('College_Completion.csv')
college_data.info()
college_cols = ["level", "control", "basic", "state"]
college_data[college_cols] = college_data[college_cols].astype('category')

# Job data
job_data = pd.read_csv('Job_Placement.csv')
job_data.info()
job_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
job_data[job_cols] = job_data[job_cols].astype('category')

# %%
## Collapse factor levels as needed
# College data
# Lambda functions are small anonymous functions useful for simple operations
college_data['control_simplified'] = college_data['control'].apply(lambda x: 'Private' if 'Private' in str(x) else 'Public').astype('category')
# College result
print(college_data['control_simplified'].value_counts())

# Job data doesn't need collapsing because the columns are already simple. 

# %%
## One-hot encoding factor variables
# College data
ccategory_list = list(college_data.select_dtypes('category'))
college_encoded = pd.get_dummies(college_data, columns=ccategory_list)
college_encoded.info()

# Job data
jcategory_list = list(job_data.select_dtypes('category'))
job_encoded = pd.get_dummies(job_data, columns=jcategory_list)
job_encoded.info()

# %%
## Normalize the continuous variables
# College data
college_numeric_cols = ['ft_pct', 'pell_value', 'retain_value', 'grad_100_value', 'awards_per_value', 'fte_value', 'aid_value', 'endow_value']
college_encoded.boxplot(column=college_numeric_cols, vert=False, grid=False)
college_encoded[college_numeric_cols].describe()

# Job data
job_numeric_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
# Used different plotting method to avoid overlapping boxplots due to the number of numeric columns and because the other one was causing issues with the display of the boxplots.
job_encoded[job_numeric_cols].plot(kind='box', vert=False, figsize=(10, 4))
job_encoded[job_numeric_cols].describe()

# %%
## Drop unneeded variables 
# College data
college_dt = college_encoded.drop(['index', 'unitid', 'chronname', 'city', 'site', 'nicknames', 'long_x', 'lat_y', 'hbcu', 'flagship', 'cohort_size', 'counted_pct', 'similar', 'grad_100_percentile', 'grad_150_percentile'], axis=1)
college_dt 

# Job data
job_dt = job_encoded.drop(['sl_no'], axis=1)
job_dt

# %%
## Create target variable if needed
# I don't think it's needed because we can use the existing variables to create a target variable.
# binary target variable for college completion based on grad_150_value (1 if above median, 0 if below)
# College data
# create binary target variable based on grad_150_value
median_grad = college_dt['grad_150_value'].median()
college_dt['high_completion'] = pd.cut(college_dt['grad_150_value'],
                                       bins=[-1, median_grad, 101],
                                       labels=[0, 1])

# Job data
# drop rows with missing salary values first
job_dt = job_dt[job_dt['salary'].notna()].copy()
# create binary target variable based on grad_150_value
median_salary = job_dt['salary'].median()
job_dt['high_salary'] = pd.cut(job_dt['salary'],
                                bins=[-1, median_salary, 1000000],
                                labels=[0, 1])

# %%
## Calculate the prevalence of the target variable
# College data
prevalence = (college_dt['high_completion'].value_counts()[1] /
              len(college_dt['high_completion']))
# value_counts()[1] gets count of '1' values (high quality)
# Divide by total count to get proportion
# Job data
prevalence_job = (job_dt['high_salary'].value_counts()[1] /
                  len(job_dt['high_salary']))
# value_counts()[1] gets count of '1' values (high quality)
# Divide by total count to get proportion

# %%
## Create the necessary data partitions (Train,Tune,Test)
# College data
college_clean = college_dt.drop(['grad_150_value', 'grad_100_value'], axis=1)
# Separate training data from the rest
college_train, college_temp = train_test_split(
    college_clean,
    train_size=0.60,
    stratify=college_clean.high_completion,
    random_state=42
)
# Split remaining data into tuning and test sets (50/50)
college_tune, college_test = train_test_split(
    college_temp,
    train_size=0.50,
    stratify=college_temp.high_completion,
    random_state=42
)

# Job data
job_clean = job_dt.drop(['salary'], axis=1)
# Separate training data from the rest
job_train, job_temp = train_test_split(
    job_clean,
    train_size=0.60,
    stratify=job_clean.high_salary,
    random_state=42
)
# Split remaining data into tuning and test sets (50/50)
job_tune, job_test = train_test_split(
    job_temp,
    train_size=0.50,
    stratify=job_temp.high_salary,
    random_state=42
)

# %% [markdown]
# ## Step 3: What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about?
# - College Completion, I believe that this data would be able to address the problem of whether there is a correlation between the level of schooling students are in currently and the graduation rate. But, I am concerned about potential confounding variables, such as socioeconomic status, access to resources, and personal motivation. I would want to ensure that the data is accurately representative of the population being studied, but these variables might skew the results.
# - Job Placement, I think that this data would be able to address the problem of whether gender has a significant impact on job salary, despite having the same qualifications. Although, I am worried about possible variables that may influence the results I am seeking. For example, experience level, industry, and location.