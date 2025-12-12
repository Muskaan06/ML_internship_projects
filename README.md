# Machine Learning Internship  

## Overview  
This repository contains the work completed during my Machine Learning Internship. The internship includes four end-to-end projects covering classical machine learning and deep learning, along with research on model training techniques such as cross-validation, hyperparameter tuning, and feature engineering.

---

## Development Setup  
- All projects were developed using **VS Code** and **Jupyter Notebook**.  
- Work was completed independently, with occasional references to online resources and ChatGPT.  
- **Projects 1–3** use general machine learning classifiers with a similar workflow.  
- **Project 4** uses a deep learning model trained on GPU.

---

## Models Studied  

### Logistic Regression  
A linear model for binary or multiclass classification using logistic or softmax activation. Works well for linearly separable data and is highly interpretable.

### Decision Tree  
A tree-structured model that splits on feature thresholds. Captures nonlinear relationships but may overfit without pruning.

### Random Forest  
An ensemble of decision trees trained on random subsets of data and features. Reduces overfitting and provides strong baseline results on tabular data.

### XGBoost  
A gradient boosting framework optimized for performance and speed, using regularization to prevent overfitting.

### LightGBM  
A fast, memory-efficient gradient boosting model based on histogram methods and leaf-wise growth. Performs well on large, high-dimensional datasets.

### ResNet50  
A 50-layer CNN with residual blocks to address vanishing gradients. Pretrained on ImageNet and widely used for transfer learning in image classification.

---

## Model Training Improvement Techniques  

### Cross-Validation  
- Splits data into multiple folds to train and evaluate across different subsets.  
- Provides more reliable generalization estimates.  
- Helps detect overfitting by comparing training and validation performance.  
- Tools: `StratifiedKFold`, `cross_val_score`.

### Hyperparameter Tuning  
- Searches for the best parameter combinations for improved accuracy and reduced overfitting.  
- Methods: `RandomizedSearchCV`, model-specific tuning for LightGBM/XGBoost.

### Feature Engineering  
- Creates, selects, or transforms features to enhance model performance.  
- Includes scaling, normalization, variance thresholding, encoding, and derived features.  
- Tools: `StandardScaler`, `PowerTransformer`, `SelectKBest`, `RFE`, `pandas`, `numpy`.

---

# Projects  

---

## Project 1: Forest-Cover Prediction  
**Problem:** Predict 7 forest cover types (multiclass classification).  
**Data:** All numerical features.  
**Preprocessing:** Scaling, train-test split, handling missing values.  
**Models Compared:** Random Forest, LightGBM, Logistic Regression, Gradient Boosting.  
**Packages:** Python, Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib.  
**Repository:** Forest-cover-prediction

### Results  
- LightGBM achieved the highest accuracy among all classifiers.

### Observations  
- Elevation and soil type were key contributing features.  
- Ensemble models outperformed linear models.

### Takeaways  
- Feature engineering had a significant impact on performance.  
- Multiclass classification requires careful model evaluation and comparison.

---

## Project 2: Heart Disease Prediction  
**Problem:** Binary classification to identify heart disease.  
**Data:** All numerical features.  
**Preprocessing:** Scaling, train-test split, handling missing values.  
**Model Used:** LightGBM.  
**Packages:** Python, Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib.  
**Repository:** Heart-Disease-Prediction

### Results  
- LightGBM achieved **94.5% accuracy** on test data.

### Observations  
- Captured nonlinear medical interactions effectively.  
- False negatives were minimized.

### Takeaways  
- Highlighted the importance of preprocessing in medical datasets.  
- Developed understanding of binary classification thresholds.

---

## Project 3: Liver Cirrhosis Stage Detection  
**Problem:** Predict liver damage stage (3-class classification).  
**Data:** Contains both categorical and numerical features.  
**Preprocessing:** Scaling, train-test split, handling missing values, `OrdinalEncoder` for categorical variables.  
**Model Used:** LightGBM.  
**Packages:** Python, Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib.  
**Repository:** Liver Cirrhosis Stage Detection

### Results  
- Achieved **96.78% accuracy** on test data.

### Observations  
- Biochemical markers had a strong predictive influence.  
- Hyperparameter tuning significantly improved accuracy.

### Takeaways  
- Learned categorical-to-numeric encoding techniques.  
- Reinforced the practical value of hyperparameter tuning.

---

## Project 4: Animal Class Classification  
**Problem:** 15-class image classification using RGB images.  
**Data:** Image dataset.  
**Preprocessing:** Resizing, random flips, rotations, normalization.  
**Model Used:** ResNet50.  
**Packages:** PyTorch, Torchvision, CUDA, Matplotlib.  
**Repository:** Animal Image Classification

### Results  
- Achieved **93% accuracy** after 10 training iterations.

### Observations  
- Transfer learning significantly boosted performance.  
- Data augmentation reduced overfitting.

### Takeaways  
- Gained experience with CNN architectures.  
- Learned GPU-based training workflows.

---

## Key Learnings  
- Importance of thorough feature engineering and preprocessing.  
- Effective use of cross-validation and hyperparameter tuning.  
- Ability to document results and conduct comparative model analysis.

---

## References  
- https://www.stepbystepdatascience.com/scikit-learn-tutorial-python-machine-learning  
- https://www.stepbystepdatascience.com/py-sklearn-pt2  
- GeeksforGeeks  
- HuggingFace  
- Medium  
- AI Tools: ChatGPT, Gemini  

