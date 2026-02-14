# %% [markdown]
# Step 1. Use the question/target variable you submitted and build a model to answer the question you created for this dataset (make sure it is a classification problem, convert if necessary).
# Question: Can we predict whether a college has a high graduation rate (above the median) based on characteristics such as level, control type, and other metrics?

# %%
# Imports - Libraries needed for data manipulation and for knn model
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

college_data = pd.read_csv('https://query.data.world/s/qpi2ltkz23yp2fcaz4jmlrskjx5qnp', encoding="cp1252")
# the encoding part here is important to properly read the data! It doesn't apply to ALL csv files read from the web,
# but it was necessary here.
college_data.info()

#%%
# NULL with np.nan for easier handling of missing data
college_data2 = college_data.replace('NULL', np.nan)

# drop columns that won't be useful for the model
drop_cols = [
    "index", "unitid", "chronname", "city", "site", "nicknames",
    "long_x", "lat_y"
]

college_data2 = college_data2.drop(columns=drop_cols, errors="ignore")

# drop variables with too many missing values
college_data2 = college_data2.drop(columns=['med_sat_value'], errors="ignore")

# convert categorical variables to category dtype
cat_cols = ["level", "control", "basic", "state"]
for col in cat_cols:
    if col in college_data2.columns:
        college_data2[col] = college_data2[col].astype("category")

# check missing percent (optional but good)
missing_percent = college_data2.isna().mean().sort_values(ascending=False)
print(missing_percent)

college_data2.info()

# %%
# clean classification target variable
college_data2['grad_150_value'] = pd.to_numeric(college_data2['grad_150_value'], errors='coerce')
college_data2['grad_150_value'].describe()
# create binary target variable based on grad_150_value
median_grad = college_data2['grad_150_value'].median()
college_data2['high_grad_rate'] = (college_data2['grad_150_value'] > median_grad).astype(int)
college_data2['high_grad_rate'].value_counts()
# drop missing values after all conversions
college_data2 = college_data2.dropna(subset=['grad_150_value'])
college_data2.info()
print(college_data2['high_grad_rate'].value_counts(normalize=True))

# %% [markdown]
# Step 2: Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning if needed changed the target variable.

# %%
X = college_data2.drop(['high_grad_rate', 'grad_150_value'], axis=1)
y = college_data2['high_grad_rate']

X = pd.get_dummies(X, drop_first=True)

# Remove missing values in predictors
X = X.dropna()
y = y.loc[X.index]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    train_size=0.60,
    stratify=y,
    random_state=42
)

X_tune, X_test, y_tune, y_test = train_test_split(
    X_temp, y_temp,
    train_size=0.50,
    stratify=y_temp,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_tune_scaled = scaler.transform(X_tune)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

# %% [markdown]
# Step 3: Create a dataframe that includes the test target values, test predicted values, and test probabilities of the positive class.

# %%
# get probabilities of positive class
y_prob = knn_model.predict_proba(X_test_scaled)[:, 1]

# results dataframe
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Prob_Positive": y_prob
})

print(results_df.head(10))

# %% [markdown]
# Step 4: No code question: If you adjusted the k hyperparameter what do you think would
# happen to the threshold function? Would the confusion matrix look the same at the same threshold 
# levels or not? Why or why not?
# - If we adjusted the k hyperparameter, it would change the sensitivity of the model to local variations in the data. A smaller k (like 1 or 3) makes the model more sensitive to noise, while a larger k smooths out predictions by considering more neighbors. This would likely affect the confusion matrix at the same threshold levels because the distribution of predicted classes could shift, leading to different counts of true positives, false positives, true negatives, and false negatives.

# %%
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# Step 5: Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
# concerns or positive elements do you have about the model as it relates to your question? 
# - My kNN model performs well at predicting whether a college has a graduation rate above the median. But my model is not perfect because it still produces false positives and false negatives. I'm concerned about the potential for overfitting with a small k value. 

# %% [markdown]
# Step 6: Create two functions: One that cleans the data & splits into training|test and one that 
# allows you to train and test the model with different k and threshold values, then use them to 
# optimize your model (test your model with several k and threshold combinations). Try not to use variable names 
# in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
#function just run them separately.) 

# %%
def clean_and_split_data(data, target_col, drop_cols=None, random_state=42):

    if drop_cols is not None:
        data = data.drop(drop_cols, axis=1, errors="ignore")

    data = data.replace('NULL', np.nan)

    # Only drop rows missing the target
    data = data.dropna(subset=[target_col])

    X = data.drop([target_col], axis=1)
    y = data[target_col]

    X = pd.get_dummies(X, drop_first=True)

    # Drop rows where predictors still have missing values
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=0.60,
        stratify=y,
        random_state=random_state
    )

    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp,
        train_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test

def train_and_test_knn(X_train, y_train, X_test, y_test, k=3, threshold=0.5):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train kNN model
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)

    # Probabilities of positive class
    y_prob = knn_model.predict_proba(X_test_scaled)[:, 1]

    # Apply custom threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Evaluate
    cm = confusion_matrix(y_test, y_pred)

    return cm, y_pred, y_prob

# %% 
# example usage of the functions
drop_columns = ['med_sat_value']

X_train, X_tune, X_test, y_train, y_tune, y_test = clean_and_split_data(
    college_data2,
    target_col='high_grad_rate',
    drop_cols=drop_columns
)

cm, y_pred, y_prob = train_and_test_knn(X_train, y_train, X_test, y_test, k=5, threshold=0.5)

print("Confusion Matrix for k=5, threshold=0.5:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# Step 7: How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 
# - The model's performance can vary based on the chosen k value. But it performed well overall. The arger k helped smooth out predictions and adjusted thresholds which impact sensitivity and specificity. 

# %% [markdown]
# Step 8: Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
# step 6. 

# %%
# Step 8: testing multiple k and threshold combinations

k_values = [1, 3, 5, 7, 9]
threshold_values = [0.3, 0.5, 0.7]

for k in k_values:
    for t in threshold_values:
        cm, y_pred, y_prob = train_and_test_knn(X_train, y_train, X_test, y_test, k=k, threshold=t)

# %%
# Step 8: Choose another target variable and build another kNN model

# Convert grad_100_value to numeric
college_data2['grad_100_value'] = pd.to_numeric(college_data2['grad_100_value'], errors='coerce')

# Create new binary target variable
median_grad_100 = college_data2['grad_100_value'].median()
college_data2['high_grad_rate_100'] = (college_data2['grad_100_value'] > median_grad_100).astype(int)

# Drop leakage columns for the new model
drop_columns_step8 = ['med_sat_value', 'grad_100_value', 'high_grad_rate', 'grad_150_value']

# Clean and split using the same function
X_train2, X_tune2, X_test2, y_train2, y_tune2, y_test2 = clean_and_split_data(
    college_data2,
    target_col='high_grad_rate_100',
    drop_cols=drop_columns_step8
)

# Train and test model for the new target
cm2, y_pred2, y_prob2 = train_and_test_knn(X_train2, y_train2, X_test2, y_test2, k=5, threshold=0.5)

print("\nConfusion Matrix for new target (high_grad_rate_100), k=5 threshold=0.5:")
print(cm2)

print("\nClassification Report for new target:")
print(classification_report(y_test2, y_pred2))