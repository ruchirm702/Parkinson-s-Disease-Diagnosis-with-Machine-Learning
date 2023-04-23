# Parkinson-s-Disease-Diagnosis-with-Machine-Learning
Abstract

Purpose of the project and target audience: The purpose of this project is to develop a predictive model that can accurately identify the presence of Parkinson's disease in individuals. The target audience for this project includes healthcare professionals, researchers, and individuals interested in early detection and treatment of Parkinson's disease.

Context of the problem and why it matters: Parkinson's disease is a degenerative disorder that affects the nervous system and can lead to significant motor and cognitive impairment. Early detection and intervention are crucial in managing the symptoms of the disease and improving the quality of life for individuals with Parkinson's. Machine learning models can help identify potential cases of Parkinson's disease and enable early diagnosis and treatment, which can significantly improve patient outcomes.

Project goals and success criteria: The goal of this project is to develop a machine learning model that can accurately predict the presence of Parkinson's disease in individuals. The success of the project will be measured by the model's performance metrics, including accuracy, precision, recall, F1 score and Roc Curve. The model should achieve high accuracy and other metrics to be considered successful in predicting Parkinson's disease accurately. Additionally, the project aims to identify the most important features contributing to Parkinson's disease's presence, providing valuable insights into the disease's pathology and potential interventions.

The Parkinson's Disease Classification dataset from the UCI Machine Learning Repository contains data on individuals with and without Parkinson's disease. The dataset has 24 features, including demographic information, medical history, and results of various medical tests. The target variable is the presence or absence of Parkinson's disease. The dataset has 195 samples, with 147 samples belonging to the negative class (no Parkinson's disease) and 48 samples belonging to the positive class (Parkinson's disease). Additional information about the dataset can be found on the UCI Machine Learning Repository website.


Conclusion

Based on the results provided, the best performing model is K-Nearest Neighbors (KNN) with an accuracy of 0.983.

Random Forest with information gain and entropy both have an accuracy of 0.966, and Decision Tree has an accuracy of 0.949. Support Vector Machine (SVM) has an accuracy of 0.915, and Logistic Regression has an accuracy of 0.797. Naive Bayes classifiers, both Gaussian (gnb) and Bernoulli (bnb), and voting classifier have an accuracy of 0.814.

The performance of the XGBoost classifier has also been provided, with an accuracy of 0.97, which is comparable to Random Forest with information gain and entropy. The precision and recall scores for both classes are high, indicating that the model is able to correctly classify both classes with high accuracy. The confusion matrix shows that only 2 of the 59 samples were misclassified by the XGBoost classifier.

Overall, based on the given results, KNN is the best performing model, closely followed by Random Forest with information gain and entropy, and XGBoost.



Model Intuition

The Parkinson's disease project aimed to develop a machine learning model that could accurately diagnose Parkinson's disease based on a set of clinical features.

In this project, we explored various classification models, including Logistic Regression, Random Forest, Support Vector Machines, and XGBoost. We also applied dimensionality reduction techniques such as Principal Component Analysis (PCA) to improve the model's performance.

After comparing the performance of these models, we found that XGBoost with PCA outperformed the other models, achieving a near-perfect AUC score of 1.00 on the test set. This suggests that the model can effectively distinguish between Parkinson's disease patients and healthy individuals based on the clinical features provided.

Overall, this project highlights the potential of machine learning techniques in aiding the diagnosis of Parkinson's disease, which can lead to earlier and more accurate detection of the disease, allowing for earlier interventions and better outcomes for patients.
