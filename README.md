# Parkinson-s-Disease-Diagnosis-with-Machine-Learning
## Abstract

### Purpose of the Project and Target Audience
The purpose of this project is to develop a **predictive model** that can accurately identify the presence of **Parkinson's disease** in individuals. The target audience for this project includes **healthcare professionals**, **researchers**, and individuals interested in the **early detection** and **treatment** of Parkinson's disease.

### Context of the Problem and Why it Matters
Parkinson's disease is a **degenerative disorder** that affects the nervous system and can lead to significant **motor** and **cognitive impairment**. Early detection and intervention are crucial in managing the symptoms of the disease and improving the quality of life for individuals with Parkinson's. Machine learning models can help identify potential cases of Parkinson's disease and enable **early diagnosis** and treatment, which can significantly improve patient outcomes.

### Project Goals and Success Criteria
The goal of this project is to develop a machine learning model that can **accurately predict** the presence of Parkinson's disease in individuals. The success of the project will be measured by the model's performance metrics, including **accuracy**, **precision**, **recall**, **F1 score**, and **ROC curve**. The model should achieve high accuracy and other metrics to be considered successful in predicting Parkinson's disease accurately. Additionally, the project aims to identify the most important features contributing to Parkinson's disease's presence, providing valuable insights into the disease's pathology and potential interventions.

## Dataset Information
The **Parkinson's Disease Classification** dataset from the **UCI Machine Learning Repository** contains data on individuals with and without Parkinson's disease. The dataset has **24 features**, including demographic information, medical history, and results of various medical tests. The target variable is the presence or absence of Parkinson's disease. The dataset includes **195 samples**, with 147 samples belonging to the negative class (no Parkinson's disease) and 48 samples belonging to the positive class (Parkinson's disease).

For additional information about the dataset, please refer to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinsons).

## Model Performance
| Model                | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|----------------------|----------|-----------|--------|----------|---------|
| K-Nearest Neighbors  | 0.983    | 0.98      | 0.98   | 0.98     | 0.99    |
| Random Forest (IG)   | 0.966    | 0.97      | 0.96   | 0.96     | 0.97    |
| Random Forest (E)    | 0.966    | 0.97      | 0.96   | 0.96     | 0.97    |
| Decision Tree        | 0.949    | 0.95      | 0.95   | 0.95     | 0.95    |
| Support Vector Machine | 0.915  | 0.92      | 0.91   | 0.91     | 0.92    |
| Logistic Regression  | 0.797    | 0.80      | 0.80   | 0.80     | 0.79    |
| Naive Bayes (GNB)    | 0.814    | 0.82      | 0.81   | 0.81     | 0.81    |
| Naive Bayes (BNB)    | 0.814    | 0.82      | 0.81   | 0.81     | 0.81    |
| Voting Classifier    | 0.814    | 0.82      | 0.81   | 0.81     | 0.81    |
| XGBoost              | 0.970    | 0.97      | 0.97   | 0.97     | 0.97    |


## Conclusion

Based on the results, the **best performing model** is **K-Nearest Neighbors (KNN)** with an accuracy of **0.983**. The following observations can be made from the model performances:
- **KNN**: Achieved the highest accuracy of 0.983.
- **Random Forest (Information Gain and Entropy)**: Both achieved an accuracy of 0.966, showcasing their robustness.
- **Decision Tree**: Provided a solid performance with an accuracy of 0.949.
- **Support Vector Machine (SVM)**: Achieved an accuracy of 0.915.
- **Logistic Regression**: Demonstrated moderate performance with an accuracy of 0.797.
- **Naive Bayes** (Gaussian and Bernoulli) and **Voting Classifier**: Both achieved an accuracy of 0.814.
- **XGBoost**: Performed very well with an accuracy of 0.970, comparable to the Random Forest models.


Based on the results provided, the best performing model is K-Nearest Neighbors (KNN) with an accuracy of 0.983.

Random Forest with information gain and entropy both have an accuracy of 0.966, and Decision Tree has an accuracy of 0.949. Support Vector Machine (SVM) has an accuracy of 0.915, and Logistic Regression has an accuracy of 0.797. Naive Bayes classifiers, both Gaussian (gnb) and Bernoulli (bnb), and voting classifier have an accuracy of 0.814.

The performance of the XGBoost classifier has also been provided, with an accuracy of 0.97, which is comparable to Random Forest with information gain and entropy. The precision and recall scores for both classes are high, indicating that the model is able to correctly classify both classes with high accuracy. The confusion matrix shows that only 2 of the 59 samples were misclassified by the XGBoost classifier.

Overall, based on the given results, KNN is the best performing model, closely followed by Random Forest with information gain and entropy, and XGBoost.

### Enhanced Conclusion
The purpose of developing a predictive model for Parkinson's disease was to enable **early detection** and improve patient outcomes through timely interventions. This project successfully demonstrated the potential of machine learning models in achieving high accuracy and reliable predictions. The **K-Nearest Neighbors (KNN)** model emerged as the top performer, providing the highest accuracy, while **Random Forest** and **XGBoost** models also showed excellent performance.

The high precision and recall scores of these models indicate their effectiveness in correctly classifying both Parkinson's and non-Parkinson's cases. This aligns with the project's goal of identifying the disease early and accurately. Furthermore, the analysis of feature importance can offer valuable insights into the disease's pathology, aiding healthcare professionals and researchers in understanding the key factors contributing to Parkinson's disease.

Overall, this project underscores the potential of machine learning techniques in revolutionizing the diagnosis of Parkinson's disease, facilitating **earlier interventions** and thereby significantly improving the quality of life for affected individuals.

## Model Intuition
The Parkinson's disease project aimed to develop a machine learning model that could accurately diagnose Parkinson's disease based on a set of clinical features.

We explored various classification models, including:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machines**
- **XGBoost**

Additionally, we applied **dimensionality reduction** techniques like **Principal Component Analysis (PCA)** to improve performance.

This project highlights the potential of machine learning in aiding Parkinson's disease diagnosis, leading to earlier and more accurate detection, and consequently, better patient outcomes.


