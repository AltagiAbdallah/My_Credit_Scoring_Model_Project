# My Credit Scoring Model Project

## Project Overview

Welcome to my Credit Scoring Model project! The main goal of this project is to **predict an individual's creditworthiness** (e.g., Good, Standard, Poor) based on their past financial behavior and other relevant personal data. I'm building a classification model that can assess risk, which is a fundamental task in financial institutions for making lending decisions.

## Approach

My approach involves using various supervised machine learning classification algorithms. I'm focusing on well-established methods like **Logistic Regression**, **Decision Trees**, and **Random Forest** to build robust predictive models.

## Key Features & Components

* **Extensive Feature Engineering**: I'm creating new, more informative features from raw financial history data to enhance model performance.
* **Comprehensive Model Evaluation**: I'm rigorously assessing my models using standard classification metrics such as **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**. I also visualize Confusion Matrices and ROC Curves.
* **Diverse Dataset**: The dataset includes crucial financial attributes like income, existing debts, payment history, credit mix, and more.

## Dataset

I'm working with two datasets: `train.csv` and `test.csv`.

* **`train.csv`**: Contains historical financial data, including the `Credit_Score` (my target variable).
* **`test.csv`**: Contains similar financial data but without the `Credit_Score`, which I'll use for final predictions or evaluation against truly unseen data.

The datasets include columns such as:
`ID`, `Customer_ID`, `Month`, `Name`, `Age`, `SSN`, `Occupation`, `Annual_Income`, `Monthly_Inhand_Salary`, `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan`, `Type_of_Loan`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Credit_Mix`, `Outstanding_Debt`, `Credit_Utilization_Ratio`, `Credit_History_Age`, `Payment_of_Min_Amount`, `Total_EMI_per_month`, `Amount_invested_monthly`, `Payment_Behaviour`, `Monthly_Balance`, and `Credit_Score` (in `train.csv`).

## Project Structure

My project is organized into distinct Python code sections, each addressing a specific phase of the machine learning pipeline. This modular approach helps with clarity, maintainability, and debugging.

The key phases are:

1.  **Data Loading & Initial Inspection**: Getting the data into Colab and a first look.
2.  **Exploratory Data Analysis (EDA)**: Deep diving into data patterns and insights.
3.  **Data Preprocessing**: Cleaning and transforming the data for modeling.
4.  **Model Training & Evaluation**: Building and assessing the performance of my classification models.

Each section is designed to be runnable independently after the preceding sections have been executed, storing processed dataframes in memory.

## Setup & How to Run

This project is designed to run seamlessly on **Google Colab**, leveraging its free GPU/TPU access (though not strictly necessary for these models) and easy integration with Google Drive.

### Prerequisites

* A Google account to access Google Colab.
* My `train.csv` and `test.csv` datasets uploaded to my Google Drive (or directly to Colab session).

### Steps to Run

1.  **Open in Google Colab**:
    * Go to [Google Colab](https://colab.research.google.com/).
    * Click on `File > Upload notebook` and upload your `.ipynb` file (or create a new one and copy the code sections).
2.  **Connect Google Drive**:
    * In the first code cell of **Section 1**, run the `drive.mount('/content/drive')` command.
    * Follow the prompts to authorize Colab to access your Google Drive.
3.  **Place Your Data**:
    * Ensure your `train.csv` and `test.csv` files are located in your Google Drive, typically in `My Drive/` or a subfolder within it. Adjust the `my_drive_base_path` variable in **Section 1.2** if your files are in a subfolder (e.g., `'/content/drive/My Drive/Credit_Data/'`).
4.  **Execute Code Cells Sequentially**:
    * Run each code cell in order, from **Section 1** through **Section 4**. Each section builds upon the previous one.
    * Pay attention to any output messages or visualizations generated.

## Detailed Project Sections

### **Section 1: Data Loading & Initial Inspection**

* **1.1 Connecting My Google Drive**: Establishes the link between Colab and my Google Drive.
* **1.2 Loading My Datasets**: Reads `train.csv` and `test.csv` into pandas DataFrames.
* **1.3 Performing Initial Data Checks**: Displays the head, info, and null counts of the raw DataFrames to get a first understanding of the data.

### **Section 2: Exploratory Data Analysis (EDA)**

* **2.1 Reviewing Basic Data Information and Missing Values**: A more detailed look at data types and a summary of missing values across the datasets.
* **2.2 Summarizing Numerical and Categorical Features**: Generates descriptive statistics for numerical columns and value counts for categorical ones.
* **2.3 Visualizing Key Distributions**: Creates histograms for numerical features and count plots for the target variable (`Credit_Score`).
* **2.4 Exploring Relationships Between Features**: Uses box plots to explore relationships between categorical and numerical features, and generates a correlation heatmap for numerical features (with robust handling for non-numeric values to ensure accurate correlations).

### **Section 3: Data Preprocessing**

* **3.1 Handling Missing Values**: Imputes missing numerical values (e.g., `Annual_Income`) with the mean and categorical values (e.g., `Credit_Mix`) with the mode.
* **3.2 Feature Engineering**:
    * Converts `Credit_History_Age` (e.g., "22 Years and 9 Months") into a single numerical feature representing total months.
    * Calculates a new `Debt_to_Income_Ratio` feature.
* **3.3 Encoding Categorical Features**:
    * Uses `LabelEncoder` to convert the `Credit_Score` target variable into a numerical format.
    * Applies `OneHotEncoder` to other nominal categorical features (`Month`, `Occupation`, `Credit_Mix`, etc.) to prepare them for machine learning algorithms.
* **3.4 Dropping Irrelevant Columns**: Removes identifier columns (`ID`, `Customer_ID`, `SSN`, `Name`) and other columns no longer needed after engineering (e.g., original `Credit_History_Age`).
* **3.5 Feature Scaling**: Applies `StandardScaler` to all numerical features, ensuring they have zero mean and unit variance, which is crucial for many models.

### **Section 4: Model Training & Evaluation**

* **4.1 Defining My Target Variable and Features**: Separates the preprocessed data into features (X) and the encoded target variable (y).
* **4.2 Splitting My Data for Training and Validation**: Divides the training data into training and validation sets (80/20 split, stratified) to ensure unbiased model evaluation.
* **4.3 My Model Training and Evaluation Function**: A reusable function that trains a given classification model, makes predictions, and calculates key metrics:
    * Accuracy
    * Precision (weighted)
    * Recall (weighted)
    * F1-Score (weighted)
    * ROC-AUC (one-vs-rest, weighted)
    * Also generates and displays Confusion Matrices and Multi-class ROC Curves.
* **4.4 Initializing and Training My Chosen Models**: Trains Logistic Regression, Decision Tree, and Random Forest classifiers using the prepared training data.
* **4.5 Comparing My Model Performance (Summary)**: Presents a clear summary table of all models' performance metrics on the validation set for easy comparison.

## Results & Insights

*(This section will be populated once you've run the models and analyzed their output. You can add points like:)*

* **Best Performing Model**: Based on my current evaluation metrics (e.g., F1-Score or ROC-AUC), Random Forest appears to be performing the strongest among the tested models.
* **Key Feature Importances**: (For tree-based models like Random Forest, you could extract and list top features that influence predictions).
* **Confusion Matrix Analysis**: (Discuss where the model is performing well and where it struggles, e.g., "The model is quite good at identifying 'Good' credit scores but sometimes confuses 'Standard' with 'Poor'").

## Future Work & Improvements

* **Hyperparameter Tuning**: Use techniques like GridSearchCV or RandomizedSearchCV to optimize the hyperparameters of the best-performing models for even better performance.
* **Advanced Feature Engineering**: Explore more complex feature interactions or external data sources if available.
* **Other Algorithms**: Experiment with more advanced classification algorithms like Gradient Boosting (XGBoost, LightGBM) or Support Vector Machines (SVMs).
* **Imbalance Handling**: If the credit score classes are heavily imbalanced, consider techniques like SMOTE or class weighting during model training.
* **Model Interpretability**: Use tools like SHAP or LIME to better understand why a model makes certain predictions.
* **Deployment**: Develop an API or web application to deploy the trained model for real-time credit scoring.

## Technologies Used

* Python 3.x
* Pandas (for data manipulation)
* NumPy (for numerical operations)
* Scikit-learn (for machine learning models and preprocessing)
* Matplotlib (for plotting)
* Seaborn (for statistical data visualization)
* Google Colab (development environment)


## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.
