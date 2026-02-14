# %% 
# define functions cluster togther parts of the cleaning you did in step 2 to each function seperate them to your discresion
# ## Step 4: Create functions for your two pipelines that produces the train and test datasets. The end result should be a series of functions that can be called to produce the train and test datasets for each of your two problems that includes all the data prep steps you took. This is essentially creating a DAG for your data prep steps. Imagine you will need to do this for multiple problems in the future so creating functions that can be reused is important. You don’t need to create one full pipeline function that does everything but rather a series of smaller functions that can be called in sequence to produce the final datasets. Use your judgement on how to break up the functions.
# - College Completion
def preprocess_college_basic(file_path):
    # Load data
    college_data = pd.read_csv(file_path)
    
    # Correct variable type/class as needed
    college_cols = ["level", "control", "basic", "state"]
    college_data[college_cols] = college_data[college_cols].astype('category')
    
    # Collapse factor levels as needed
    college_data['control_simplified'] = college_data['control'].apply(
        lambda x: 'Private' if 'Private' in str(x) else 'Public'
    ).astype('category')
    
    # Drop unneeded variables
    college_dt = college_data.drop(
        ['index', 'unitid', 'chronname', 'city', 'site', 'nicknames',
         'long_x', 'lat_y', 'hbcu', 'flagship', 'cohort_size',
         'counted_pct', 'similar', 'grad_100_percentile',
         'grad_150_percentile'],
        axis=1
    )
 
    return college_dt


def encode_and_visualize_college(college_dt):
    # One-hot encoding factor variables
    ccategory_list = list(college_dt.select_dtypes('category'))
    college_encoded = pd.get_dummies(college_dt, columns=ccategory_list)
    
    # Normalize the continuous variables
    college_numeric_cols = ['ft_pct', 'pell_value', 'retain_value', 
                            'grad_100_value', 'awards_per_value', 'fte_value', 
                            'aid_value', 'endow_value']
    
    college_encoded.boxplot(column=college_numeric_cols, vert=False, grid=False)

    scaler_college = StandardScaler()
    college_encoded[college_numeric_cols] = scaler_college.fit_transform(
        college_encoded[college_numeric_cols]
    )

    return college_encoded


def create_college_targets_and_split(college_encoded):
    # Create target variable
    median_grad = college_encoded['grad_150_value'].median()
    college_encoded['high_completion'] = pd.cut(
        college_encoded['grad_150_value'],
        bins=[-1, median_grad, 101],
        labels=[0, 1]
    )
    
    # Calculate prevalence
    prevalence = college_encoded['high_completion'].astype(int).mean()
    
    # Create partitions
    college_clean = college_encoded.drop(['grad_150_value', 'grad_100_value'], axis=1)
    
    college_train, college_temp = train_test_split(
        college_clean,
        train_size=0.60,
        stratify=college_clean.high_completion,
        random_state=42
    )
    
    college_tune, college_test = train_test_split(
        college_temp,
        train_size=0.50,
        stratify=college_temp.high_completion,
        random_state=42
    )
    
    return college_train, college_tune, college_test, prevalence


# - Job Placement
def preprocess_job_basic(file_path):
    # Load data
    job_data = pd.read_csv(file_path)

    # Save status separately as the TARGET to avoid leakage
    job_target = job_data['status']

    # Drop status from predictors (leakage issue)
    job_data = job_data.drop(['status'], axis=1)

    # Correct variable type/class as needed
    job_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
    job_data[job_cols] = job_data[job_cols].astype('category')

    return job_data, job_target


def encode_and_visualize_job(job_data, job_target):
    # One-hot encoding factor variables
    jcategory_list = list(job_data.select_dtypes('category'))
    job_encoded = pd.get_dummies(job_data, columns=jcategory_list)

    # Create target variable (Placed = 1, Not Placed = 0)
    job_encoded['placed'] = (job_target == "Placed").astype(int)

    # Normalize the continuous variables
    job_numeric_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
    job_encoded[job_numeric_cols].plot(kind='box', vert=False, figsize=(10, 4))

    scaler_job = StandardScaler()
    job_encoded[job_numeric_cols] = scaler_job.fit_transform(job_encoded[job_numeric_cols])

    # Drop unneeded variables (salary has structural missingness tied to placement)
    job_encoded = job_encoded.drop(['sl_no', 'salary'], axis=1)

    return job_encoded


def create_job_targets_and_split(job_encoded):
    # Calculate prevalence of placement
    prevalence_job = job_encoded['placed'].mean()

    # Create partitions
    job_clean = job_encoded.copy()

    job_train, job_temp = train_test_split(
        job_clean,
        train_size=0.60,
        stratify=job_clean.placed,
        random_state=42
    )

    job_tune, job_test = train_test_split(
        job_temp,
        train_size=0.50,
        stratify=job_temp.placed,
        random_state=42
    )

    return job_train, job_tune, job_test, prevalence_job

