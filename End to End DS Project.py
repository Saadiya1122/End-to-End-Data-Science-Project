#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align: center;">Credit Card Fraud Detection 2023</h1>

# # 1. Abstract
# 
# In this research, we aim to address the burgeoning issue of credit card fraud in the digital age. Using an extensive set of more than 550,000 credit card transactions made by European cardholders in 2023, we are concentrating on applying cutting-edge AI and machine learning methods to create an effective fraud detection system. The scope of our study includes a thorough examination of the literature, an in-depth investigation of key features in the dataset, the development of relevant research questions, an effective methodology, and an objective evaluation of many machine learning models.

# # 2. Introduction
# 
# Credit card fraud, an increasingly formidable challenge in today's dynamic digital landscape, poses significant threats to both financial institutions and consumers alike. 
# The growing number of electronic transactions has made it necessary to develop and apply creative solutions that can effectively block the constantly evolving tactics used by fraudsters. The significant financial consequences and imminent risk to customer trust are sufficient to highlight how serious this issue is. Deploying and establishing improved fraud detection methods that can effectively identify and reduce the dangers associated with illegal financial transactions is imperative considering these challenges.

# # 3. Literature Review
# 
# The detection of credit card fraud has been a recurring issue in the field of financial security, and researchers have thoroughly investigated several approaches to address this dynamic problem. A thorough analysis of this field of literature indicates the intricacy of credit card theft and the demand for advanced detection technologies.
# 
# #### Machine Learning Approaches:
# Numerous studies have delved into the application of machine learning algorithms for fraud detection. Vaishnavi Nath Dornadula et al. (2020) emphasized the efficacy of machine learning in their research on credit card fraud detection. They explored diverse algorithms, highlighting the importance of algorithm selection in achieving high detection accuracy.
# 
# #### Anomaly Detection Techniques:
# Anomaly detection has emerged as a pivotal approach in identifying fraudulent transactions. Meenu et al. (2020) conducted research specifically on anomaly detection in credit card transactions using machine learning. Their work shed light on the effectiveness of anomaly detection techniques in capturing irregular patterns indicative of fraud.
# 
# #### Integration of Multiple Techniques:
# The landscape of credit card fraud is dynamic, requiring a holistic approach. Research by Btoush et al. (2023) emphasizes the importance of integrating multiple techniques, such as machine learning algorithms, anomaly detection, and deep learning, to create robust and adaptive fraud detection systems.
# 
# #### Gap Analysis:
# While existing studies provide valuable insights, there is a recognized gap in the literature. The need for improved precision and adaptability in fraud detection systems is highlighted, aligning with the objectives of the current research.
# 
# In conclusion, the assessment of the literature highlights the variety of methods used in credit card fraud detection, from traditional machine learning algorithms to more advanced deep learning approaches. Developing efficient and flexible fraud detection systems that can handle the changing strategies used by fraudsters in the digital era requires integrating different methods.

# # 4. Dataset Features
# 1 - The dataset includes credit card transactions in 2023 carried out by cardholders across Europe.
# 
# 2 - It contains more than 550,000 records with anonymised transaction characteristics, including the time and location of the transaction as well as several features (V1 to V28).
# 
# 3 - A binary label ("Class") indicating whether the transaction is fraudulent (1) or not (0) also has recorded, along with the transaction value.

# # 5. Research Questions
# 
# Our exploration is guided by a set of pertinent research questions, steering the investigation towards practical solutions:
# 
# 
# 5.1 - What are the key features or indicators crucial for identifying fraudulent transactions?
# 
# 5.2 - What are the potential implications for customer service and communication when a fraudulent transaction is detected?
# 
# 5.3 - How can innovative approaches like deep learning and anomaly detection enhance the precision of fraud detection?
# 
# 5.4 - What are the most common fraud-related transaction categories, and how can business tactics be modified to counteract these risks?
# 
# 5.5 - How can organizations adapt to emerging fraud tactics over time to maintain the effectiveness of fraud detection models?

# # 6. Methodology

# ## 6.1 Data Collection

# Over 550,000 records have been collected, all of which conceal the details of credit card transactions made by European cardholders in 2023. The source of this invaluable dataset is attributed to Kaggle.

# ## 6.2 Data Exploration
# 
# We thoroughly reviewed the dataset, gathering important information and calculating thorough statistics with the help of the info() and describe() methods. We thoroughly investigated the column data types to ensure that they were compatible with machine learning models. The class distribution was presented visually using a countplot, which let us distinguish between authentic and fraudulent transactions. The dataset's integrity was thoroughly checked for missing values, providing an accurate starting point for further analysis and model building.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("creditcard_2023.csv")
df.head()


# Checking the shape of the dataset.

# In[3]:


df.shape


# Display basic information about the dataset using .info()
# 
# #### Output:
# The database possesses 31 columns and 568,630 assets that reflect different credit card transaction information. These characteristics include the target variable ("Class"), which determines whether a transaction is fraudulent (Class=1) or not (Class=0), anonymized features (V1 to V28), and transaction amounts ("Amount"). There are no missing values in any of the columns, suggesting that the data is completely non-null. The dataset takes up around 134.5 MB of RAM. Two columns are of type int64, which is used for discrete integer values, while the remaining columns are of type float64, which represents numerical data. comprehending and becoming ready for additional analysis of the data requires having an in-depth knowledge of the dataset's structure and data types.

# In[4]:


df.info()


# Using .describe() for Summary statistics
# 
# #### Output:
# The dataset's distribution and attributes are shown by the summary statistics, which show that features V1 to V28 have been standardized and concealed with a mean close to 0 and a standard deviation close to 1. The transaction amounts represented by the "Amount" feature have an estimated mean of 12,041.96 and a standard deviation of 6,919.64. With a mean of 0.5, the "Class" column, which differentiates fraudulent (Class=1) from non-fraudulent (Class=0) transactions, indicates a rather balanced distribution. An in-depth understanding of feature qualities is provided by additional statistics, such as minimum, maximum, and percentile values, which aid in gathering the data for further study.

# In[5]:


df.describe()


# Checking the data types

# In[6]:


df.dtypes


# Below we are checking the 'Class' distribution for fraudulent and Non-fraudulent transactions.
# 
# #### Output:
# The output shows that we have a balanced dataset with an equal number of fraudulent (Class 1) and non-fraudulent (Class 0) transactions, each having 284,315 records.

# In[7]:


# Class distribution (fraudulent and non-fraudulent transactions)
print(df["Class"].value_counts())


# A countplot was used to show the distribution of fraudulent (Class=1) and non-fraudulent (Class=0) transactions.

# In[8]:


# Visualize the class distribution
sns.countplot(data=df, x='Class')
plt.title("Class Distribution (0: Non-Fraudulent, 1: Fraudulent)")
plt.show()


# In the initial phase of data exploration, we performed a comprehensive check for missing values within the dataset. Our analysis revealed that there are no missing values in any of the columns, confirming the completeness and integrity of the dataset. This is a pivotal finding as it ensures that the dataset is suitable for further analysis and model development. 

# In[9]:


# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)


# The dataset has been cleaned for analysis. The next stage is to determine which columns are necessary for our study and perhaps eliminate those that are not. We can better concentrate on appropriate features for our machine learning models by simplifying this approach.

# ## 6.3 Data Cleaning

# The "id" column was dropped from the dataset.

# In[10]:


df = df.drop('id', axis = 1) # dropping the id column


# Using the Z-score approach, outliers in the dataset were identified. Every data point's Z-score was computed using the detect_outliers_zscore function, and outliers were identified using a threshold of 3. Outlier-filled rows have been identified and shown. These anomalies will be taken into account in the stages of our study that follow if they have an influence on the analysis.
# 
# #### Output:
# The DataFrame indicates that no rows with outliers were found in this study. This implies that there are no significant numerical outliers in the dataset.

# ## 6.4 Outlier Detection

# In[11]:


from scipy import stats

# Define a function to detect and remove outliers based on Z-score
def detect_outliers_zscore(df, threshold=3):
    z_scores = np.abs(stats.zscore(df))
    outliers = (z_scores > threshold).all(axis=1)
    return outliers

# Apply the function to the dataset (excluding non-numeric columns like "Class")
numeric_columns = df.drop(["Class"], axis=1)
outliers = detect_outliers_zscore(numeric_columns)

# Show the rows with outliers
outlier_rows = df[outliers]
print("Rows with outliers:\n")
print(outlier_rows)


# ## 6.5 Box Plot for Amount

# Plotly Express was used to create a box plot that showed the distribution of transaction amounts by class (fraudulent and non-fraudulent). For every class, the plot displays the distribution, the central tendency, and any possible outliers. Box plots are used to display the values for the 'Amount' for both classes. The interquartile range (IQR) for each class is displayed in a box, and a line inside the box represents the median. Individual data points outside of this range are regarded as outliers. The whiskers extend to the lowest and greatest values within a specified range. To differentiate between transactions that are fraudulent (Class=1) and those that are not (Class=0), the 'Class' variable is utilized as the x-axis. The 'points="all"' argument in the box plot configuration ensures that individual data points are displayed as markers, making it easy to identify potential outliers. The size of the markers has been adjusted to three for the purpose of increasing their visibility. Plotting gives the distribution of transaction amounts across the two groups a visual comparison, which is essential for spotting variations that may help in the identification of fraud.
# 
# #### Output:
# The box plot analysis shows that although the range and variability of transaction amounts are identical, there are slight variations in the central tendency of transaction amounts between fraudulent and non-fraudulent transactions. This suggests that additional variables or analysis are required to improve fraud detection accuracy and that transaction quantity alone may not be an accurate indicator of fraud.

# In[12]:


import plotly.express as px

# Create a box plot for the 'Amount' feature
fig = px.box(df, x='Class', y='Amount', points="all", title="Box Plot of Amount by Class")
fig.update_traces(marker=dict(size=3))  # Adjust the marker size for outlier points
fig.show()


# ## 6.6 Correlation Matrix

# In the below code a correlation matrix is calculated using the .corr() method on a DataFrame (df). All of the DataFrame's numerical columns' pairwise correlation coefficients are calculated using this approach. The correlation_matrix that is produced is square in shape, with every element indicating the correlation coefficient between two columns.
# 
# #### Output:
# The relationship between characteristics and the "Class" column is particularly relevant when it comes to fraud detection. The following are some significant findings from the correlation matrix:
# 
# 
# - The "Class" column displays significantly large positive correlations with features V17, V14, V12, V10, V11, V4, V2, V7, and V19, suggesting that these features may be more important for identifying fraudulent transactions.
# 
# 
# - The "Class" column shows significant negative correlations with features V3, V16, V1, V6, V9, V18, V5, and V21, indicating an inverse relationship between these parameters and fraudulent transactions.
# 
# 
# - The 'Amount' feature shows a very low correlation with the 'Class' feature, this indicated that it might not be a strong indicator of fraud on an individual basis.

# In[13]:


# Calculate and display the correlation matrix
correlation_matrix = df.corr()
correlation_matrix


# Visualising the above correlation matrix using seaborn library.
# The code generates a heatmap that shows the relationships between the dataset's different features. The correlations are displayed by colors that indicate their strength and direction (positive or negative), and the correlation coefficients are given precisely by the numerical values in each cell. 

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Set up a larger matplotlib figure
plt.figure(figsize=(20, 15))

# Create a heatmap using seaborn with annotated correlation coefficients
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# Set the title of the heatmap
plt.title("Correlation Heatmap")

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the heatmap
plt.show()


# ## 6.7 Model Development

# The code below imports the necessary libraries, such as One-Class SVM for anomaly detection, Random Forest Classifier, and Logistic Regression, before preparing the data for the machine learning classification task. Additionally, it imports the classification_report function, which is then used to assess the model's effectiveness. Parallel and delayed from the joblib library were also imported as these are used for parallel processing, which can be useful for more efficient execution, especially for computationally intensive tasks. The dataset is then split into features (X) and the target variable (y) for a classification job, where the target class labels are represented by 'y' and the data features by 'X'. These will be used for the evaluation and training of classification models.

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
from joblib import Parallel, delayed

# Split the data into features (X) and the target variable (y)
X = df.drop('Class', axis=1)
y = df['Class']


# The code is a binary classification procedure for machine learning that identifies fraudulent detection. 
# 
# 
# 1 - Data Preparation:
# First, the dataset is split into the target variable (y) and its features (X). 'Class,' the target variable, which specifies whether a transaction is fraudulent (1) or not (0).
# 
# 
# 
# 2 - Data Split:
# The dataset was then split into training and testing using the 'train_test_split' function from scikit learn. Here, the dataset is split into 80% and 20% for training and testing respectively.
# 
# 
# 
# 3 - Model Initialization:
# 
# There are three initialised machine learning models:
# 
# - lr_model: A popular linear classification approach is called logistic regression.
# 
# 
# - A decision tree-based ensemble learning technique is called Random Forest (rf_model).
# 
# 
# - OneClassSVM (svm_model): Novelty detection using a support vector machine model. Here, it's applied to detect abnormalities or outliers in the data.
# 
# 
# 4 -  Model Training:
# The training i.e X_train and y_train is used to train on each models. The lr_model, rf_model, svm_model were used to train the data respectively. Then the train data were saved using joblib as this allows for easy reuse of the models without the need to retrain them in the future.
# 
# 
# 5 - Model Predictions:
# The three trained models (lr_model, rf_model, and svm_model) are used to make predictions on the test data (X_test).
# 
# 
# 6 -  Evaluation Metrics:
# Based on the  test data and the model's predictions, the algorithm calculates and generates the evaluation metrics listed below for each model:
# 
# 
# - Accuracy: Calculates the amount of accurate forecasts.
# 
# 
# - Precision: Evaluate how well the model can anticipate favorable outcomes.
# 
# 
# - F1 Score: An equilibrium between recall and accuracy that is particularly helpful in handling unbalanced datasets.
# 
# #### Output:
# - The Random Forest model has a very high F1 score, accuracy, and precision, and works very effectively. It indicates that fraudulent transaction detection is an attribute of the Random Forest model.
# 
# - High accuracy, precision, and F1 scores are also achieved using the Logistic Regression model, which also performs well. It's a reliable choice for this study.
# 
# - In contrast, the SVM model (OneClassSVM) has inadequate performance. Its considerably lower F1 score, accuracy, and precision imply that it would not be appropriate for this particular fraud detection task.
# 
# Based on the evaluation metrics provided, the Random Forest model appears to be the best choice for the task of fraud detection in this study. It has demonstrated high accuracy, precision, and F1 score, indicating its ability to effectively detect fraudulent transactions.

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
import joblib

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100)
svm_model = OneClassSVM(nu=0.05)

# Train the models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svm_model.fit(X_train)

# Save the trained models using Joblib
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(svm_model, 'svm_model.joblib')

# Make predictions
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)


threshold = 0  # Define a suitable threshold
svm_predictions = (svm_model.predict(X_test) < threshold).astype(int)

# Evaluate the models
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Precision:", precision_score(y_test, lr_predictions))
print("F1 Score:", f1_score(y_test, lr_predictions))
print("----------------------------------------------")

print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Precision:", precision_score(y_test, rf_predictions))
print("F1 Score:", f1_score(y_test, rf_predictions))
print("----------------------------------------------")

print("SVM:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Precision:", precision_score(y_test, svm_predictions))
print("F1 Score:", f1_score(y_test, svm_predictions))


# ## 6.8 Anomaly Detection

# The below code is performing anomaly detection using a Random Forest classifier.
# 
# 1 - Importing necessary libraries:
# - RandomForestClassifier from scikit-learn's ensemble module for building a Random Forest model.
# - train_test_split for splitting the dataset into training and testing sets.
# - precision_score, recall_score, and f1_score from scikit-learn's metrics module for evaluating the model.
# 
# 2 - Data split:
# The dataset (X, features, and y, target variable) is split into training (80%) and testing (20%) sets using train_test_split.
# 
# 3 - Initialize and train Random Forest Model:
# A RandomForestClassifier object (rf_model) with 100 trees is generated. Fit is then used to train the model using the training set of data.
# 
# 4 - Make Predictions:
# Predict is used to make predictions based on the testing results. The predictions in this instance are binary (0 for true transactions and 1 for fraudulent ones).
# 
# 5 - Set a threshold:
# A threshold value (in this case, 0.5) is used to identify anomalies. The probability at which a transaction can be considered fraudulent is defined by this threshold. 
# 
# 6 - Evaluate the model:
# The accuracy, recall, and F1 score of the anomaly detection model are computed to evaluate its performance. These metrics assist in evaluating the model's overall efficacy, accuracy, and completeness in detecting fraudulent transactions.
# 
# 7 - Print the evaluation results:
# The code prints out the precision, recall, and F1 score for the anomaly detection performed with the Random Forest model.
# 
# #### Output:
# To summarise, the Random Forest model shows remarkable performance in anomaly identification, as demonstrated by its high accuracy, recall, and F1 score. This model is a great option for anomaly detection since it is very good at detecting fraudulent transactions while reducing false alarms.

# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Make predictions (0 for normal, 1 for fraudulent)
rf_predictions = rf_model.predict(X_test)

# Determine a threshold for classifying anomalies
threshold = 0.5

# Classify transactions based on the threshold
anomalies = (rf_model.predict_proba(X_test)[:, 1] > threshold).astype(int)

# Evaluate the model for anomaly detection
precision = precision_score(y_test, anomalies)
recall = recall_score(y_test, anomalies)
f1 = f1_score(y_test, anomalies)

print("Anomaly Detection (Random Forest):\n")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# The trained Random Forest model was saved to a file using Joblib for future use.

# In[18]:


# Save the trained model to a file using joblib
joblib.dump(rf_model, 'fraud_detection_model.pkl')


# ## 6.9 Real-time Predictions
# 
# In order to identify fraud, the  code imports a trained Random Forest model and applies it to categorize newly received transaction data as either legitimate or fraudulent (Here we are using some random numbers as an example). The model predicts a binary classification and accepts input from numerous features (V1 to V28 and Amount).
# 
# The code begins by loading the trained Random Forest model using the joblib.load function then a new Pandas DataFrame containing feature values like 'V1', 'V2',..., 'Amount' is prepared. The loaded model is used to make predictions on the new data using the predict method. A threshold point (0.5) has been set for classifying anomalies. This threshold is used for comparing the model's predictions. The predicted amount is classified as a fraudulent transaction and the appropriate response is performed (e.g., alert, block, or investigate) if it exceeds the threshold (1). The transaction will be processed as a regular transaction if the forecast is less than or equal to the threshold (0).
# This code can be integrated into a larger system for real-time fraud detection or batch processing of transaction data.
# 
# #### Output:
# The code output "Normal Transaction," which means that the provided transaction data was classified as a normal transaction by the loaded Random Forest model for fraud detection. This indicates that the model did not flag the transaction as fraudulent based on the input features and the chosen threshold.

# In[19]:


# Load the trained Random Forest model
loaded_model = joblib.load('fraud_detection_model.pkl')

new_data = pd.DataFrame({
    'V1': [0.5],
    'V2': [-0.3],
    'V3': [1.2],
    'V4': [-0.7],
    'V5': [0.4],
    'V6': [0.8],
    'V7': [0.6],
    'V8': [-0.2],
    'V9': [0.5],
    'V10': [0.7],
    'V11': [0.1],
    'V12': [0.2],
    'V13': [0.3],
    'V14': [0.4],
    'V15': [0.5],
    'V16': [0.6],
    'V17': [0.7],
    'V18': [0.8],
    'V19': [0.9],
    'V20': [0.10],
    'V21': [0.11],
    'V22': [0.12],
    'V23': [0.13],
    'V24': [0.14],
    'V25': [0.15],
    'V26': [0.16],
    'V27': [0.17],
    'V28': [0.18],
    'Amount': [100.0]
})


# Make predictions on the new data using the trained model
predictions = loaded_model.predict(new_data)

# Define a threshold for classifying anomalies
threshold = 0.5

# Classify transactions based on the threshold
classified = (predictions > threshold).astype(int)

# Depending on your business logic, you can take various actions based on the classification results:
if classified == 1:
    # Take actions for fraudulent transactions (e.g., alert, block, or investigate)
    print("Fraudulent Transaction Detected!")
else:
    # Process normal transactions
    print("Normal Transaction")


# # 7. Results

# In our comprehensive analysis of the credit card fraud detection dataset, various aspects were explored, leading to the development and evaluation of machine learning models for fraud detection.
# 
# ### Explorating Data Analysis:
# 
# - Dataset Summary: The dataset, comprising 568,630 entries, exhibited no missing values, providing a solid foundation for subsequent analyses.
# 
# - Visualization: Utilizing a box plot, we observed higher amounts in fraudulent transactions, emphasizing their distinctive nature.
# 
# - Statistical Analysis: Correlation matrix analysis identified features with strong correlations, particularly 'V14' and 'V17,' indicating their significance in fraud identification.
# 
# - Outlier Detection: Although an outlier detection method based on Z-scores was applied, no outliers were found.
# 
# ### Model Building and Evaluation:
# 
# - Data Split: The dataset was divided into 80% training and 20% testing sets.
# 
# - Models Trained: One-Class SVM, Random Forest, and Logistic Regression were the three models that were trained.
# 
# - Model Evaluation: Compared to the other models, the Random Forest model performed better, obtaining a high F1 score, accuracy, and precision. The model that was found to be most appropriate for detecting fraud was accepted.
# 
# - Anomaly detection: The Random Forest model was used to discover anomalies, and a threshold for classifying them was determined. The model had a strong ability to detect fraudulent transactions, as indicated by its excellent accuracy, recall, and F1 score.
# 
# ### Answers to Research Questions:
# 
# #### 5.1 - What are the key features or indicators that can be used in the identification of fraudulent transactions?
# 
# - The dataset analysis revealed that certain features, such as 'V1' through 'V28' and 'Amount,' are crucial for identifying fraudulent transactions. Notably, features having the strongest negative correlations with the 'Class' variable are 'V14' and 'V17,' which can be valuable indications. Another way to increase feature relevance is to study feature engineering and selection methods.
# 
# #### 5.2 - What are the potential implications for customer service and communication when a fraudulent transaction is detected?
# 
# - Better customer service can result from detecting fraudulent transactions as they stop unauthorized transactions and guarantee safety for customers. It's critical to notify customers as soon as a fraudulent transaction becomes apparent in order to explain the circumstances, walk them through the next steps, and offer help. Reducing possible interruptions and upholding confidence are two benefits of having an efficient customer communication strategy.
# 
# #### 5.3 - Are there any innovative approaches or technologies, such as deep learning or anomaly detection, that could improve the precision of fraud detection in the future?
# 
# - Indeed, cutting-edge technologies with the potential to improve fraud detection precision include deep learning and anomaly detection. Neural networks and other deep learning algorithms are able to recognize intricate patterns in data and adjust to changing fraud strategies. Through the reduction of false positives and the identification of new fraud trends, anomaly detection techniques—especially when integrated with dynamic thresholding—can improve detection precision.
# 
# #### 5.4 - What are the most regular fraud-related transaction categories in the dataset, and how can the business modify its fraud protection tactics to counteract these particular risks?
# 
# - Transaction types are not specifically classified in the dataset. Transaction data would need to be analyzed or classified in order to determine the most common fraud-related categories. This might reveal which kinds of transactions are more frequently the target of fraud. Changing fraud prevention strategies would entail focusing on the weaknesses connected to these kinds of high-risk transactions.
# 
# #### 5.5 - In order to maintain the effectiveness of the fraud detection models over time, how can the organization adapt to emerging fraud tactics and techniques?
# 
# - The following actions should be taken by the organization to respond to new fraud strategies and techniques:
# 
# 
#   a) Use dynamic thresholding to continually change model sensitivity.
# 
# 
#   b) Retrain and update models on a regular basis with new data to take changing strategies into consideration.
# 
# 
#   c) Explore adding new features and cutting-edge methods to expand the feature set.
# 
# 
#   d) To remain ahead of new risks and study advanced fraud detection technology, make research and development investments.
#   
# 
# In the context of evolving fraud methods, these tactics will assist the organisation in continuing to detect fraud effectively.

# # 8. Limitations
# 
# Recognizing the limits is essential even if our study yielded insightful results. Specific fraud-related categories are difficult to identify in the dataset due to the absence of clear transaction type classification. Furthermore, it highlights the dynamic nature of fraud detection and the need for constant effort to adapt to new fraud strategies.

# # 9. Outlooks
# 
# In order to improve the accuracy of fraud detection, further studies may involve methods based on deep learning, feature engineering, and dynamic thresholding. It should continue to be a priority to update models often and implement preventative measures to counter new fraud techniques.

# # 10. Conclusion
# Our study aims to avoid fraud with credit cards by developing and evaluating machine learning models. The best-performing model was the Random Forest model, highlighting the significance of model selection. Model implementation, dynamic thresholding, continuous improvements, feature improvement, and proactive communication with consumers are among the suggestions. Organizations need to be on the lookout for emerging fraud strategies and make continuous investments in research and development.

# # References

# N.E., 2023, Credit Card Fraud Detection Dataset 2023, Kaggle. Available at: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023 (Accessed: 05 November 2023).
# 
# Learn, no date, scikit. Available at: https://scikit-learn.org/stable/index.html (Accessed: 05 November 2023).
# 
# Joblib, no date, Running python functions as pipeline jobs. Available at: https://joblib.readthedocs.io/en/latest/ (Accessed: 05 November 2023).
# 
# Vaishnavi Nath Dornadula et al., 2020, Credit card fraud detection using machine learning algorithms, Procedia Computer Science. Available at: https://www.sciencedirect.com/science/article/pii/S187705092030065X (Accessed: 05 November 2023).
# 
# Meenu et al., 2020, Anomaly detection in credit card transactions using Machine Learning, SSRN. Available at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3670230 (Accessed: 05 November 2023).
# 
# Marazqah Btoush, E.A.L. et al., 2023, A systematic review of literature on credit card cyber fraud detection using machine and Deep Learning, PeerJ. Computer science. Available at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10280638/ (Accessed: 05 November 2023).
# 
# Btoush, E.A.L. et al., 2023, "A systematic review of literature on credit card cyber fraud detection using machine and Deep Learning," PeerJ. Computer science. Available at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10280638/ (Accessed: 05 November 2023).

# In[ ]:




