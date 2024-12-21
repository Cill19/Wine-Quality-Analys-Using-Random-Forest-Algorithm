# 🍷 Wine Quality Analysis using Random Forest

## 🌟 Overview
Welcome to the **Wine Quality Analysis Project**! This project delves into understanding what makes a wine "good" or "bad" using data from physicochemical tests and sensory evaluations. We employ cutting-edge machine learning techniques, specifically the **Random Forest algorithm**, to uncover the secrets behind wine quality.

### 🏆 Objectives
- 📊 **Explore and visualize** the wine quality dataset.
- 🛠️ **Preprocess** the data to handle class imbalance and scale numerical features.
- 🤖 **Train and evaluate** a Random Forest model to predict wine quality.
- 🧠 **Optimize** the model through hyperparameter tuning.
- 🔍 **Analyze feature importance** to identify key contributors to wine quality.

---

## 📚 Dataset Information

### 📥 Source
The dataset is sourced from the UCI Machine Learning Repository. You can find it here: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).

### 🧾 Features
- **Input Variables:**
  - Fixed Acidity
  - Volatile Acidity
  - Citric Acid
  - Residual Sugar
  - Chlorides
  - Free Sulfur Dioxide
  - Total Sulfur Dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Output Variable:**
  - Quality (Score from 0 to 10, later categorized as Low, Medium, or High)

### 🔍 Characteristics
- **Instances:** 4,898 (red and white wine samples)
- **Features:** 11 physicochemical attributes
- **Target:** Quality score
- **Missing Values:** None

---

## 📂 Project Structure
```
wine-quality-analysis/
├── data/
│   ├── raw/                # Raw dataset
│   ├── processed/          # Processed dataset after preprocessing
├── notebooks/
│   ├── data_exploration.ipynb  # Data exploration and visualization
│   ├── model_training.ipynb    # Model training and evaluation
│   ├── feature_analysis.ipynb  # Feature importance analysis
├── src/
│   ├── preprocessing.py    # Preprocessing scripts
│   ├── model.py            # Model definition and utilities
├── README.md               # Project overview and instructions
├── requirements.txt        # Dependencies
└── LICENSE                 # License information
```

---

## 📈 Key Results

### 🏅 Performance Metrics
- **Accuracy:** Achieved an accuracy of 92% on the test set after hyperparameter tuning.
- **Precision & Recall:** Provided a detailed breakdown of precision, recall, and F1-scores for each quality category.
- **Confusion Matrix:** Visualized to identify areas of misclassification.

### 🔑 Feature Importance
Top contributors to wine quality:
- **Alcohol**: Higher levels often correlate with better quality.
- **Density**: A critical indicator influenced by sugar and alcohol content.
- **Sulphates**: Impact wine stability and flavor.
- **Citric Acid**: Adds freshness and enhances the overall flavor profile.

### 📊 Visualizations
- **Feature Distributions**: Histograms and boxplots for feature-target relationships.
  ![image](https://github.com/user-attachments/assets/8062c2cc-af98-4fe8-bcc5-3bd4a0241371)

**Corelation HeatMap**: To explore inter-feature relationships.
![image](https://github.com/user-attachments/assets/6cde173f-1123-4bc8-ba11-2e5c010ad1ce)

- **Confussion Matrix**: To explore inter-feature relationships.
  ![image](https://github.com/user-attachments/assets/231e6d57-f02d-4be0-bd8b-59d84fee2ed3)
  ![image](https://github.com/user-attachments/assets/3b3c0cf3-27d4-470d-a1a3-decaa9577f87)

  - **Feature Importance Rankings**: Bar plots and cumulative importance curves.
  ![image](https://github.com/user-attachments/assets/71aa1004-6d48-4220-87c5-a38aeb0f380f)
  ![image](https://github.com/user-attachments/assets/ce3f2f3e-691e-465e-89ba-ce2fdd9eb456)

- **Sample Test**:
  ![image](https://github.com/user-attachments/assets/4b9aaa00-c36a-449c-a73a-728bdca26df6)


---

## 🚀 How to Run

### 🛠️ Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd wine-quality-analysis
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Fetch the Dataset**:
   Download it directly from the UCI repository or use the `ucimlrepo` library to automate the process.

### 🧑‍💻 Run the Project
1. **Explore the Dataset**:
   - Open `notebooks/data_exploration.ipynb`.
   - Run the notebook to visualize and explore data.
2. **Train the Model**:
   - Open `notebooks/model_training.ipynb`.
   - Train the Random Forest model and evaluate its performance.
3. **Analyze Features**:
   - Open `notebooks/feature_analysis.ipynb`.
   - Identify the most important features impacting wine quality.

### ⚙️ Customization
- Adjust hyperparameters in `notebooks/model_training.ipynb`.
- Modify feature bins in `notebooks/data_exploration.ipynb`.

---

## 🎨 Interactivity
- 🛠️ **Customize thresholds**: Adjust bins for quality categories (Low, Medium, High).
- 🔍 **Experiment with features**: Select subsets to train the model and observe changes.
- ⚙️ **Tune hyperparameters**: Modify the GridSearchCV settings for further optimization.

---

## 📦 Dependencies
- Python 3.x
- Pandas
- NumPy
- Seaborn
- Matplotlib
- scikit-learn
- imbalanced-learn
- ucimlrepo

---

## 🔮 Future Work
- 🧠 **Test other algorithms**: Experiment with Gradient Boosting, XGBoost, or neural networks.
- 🧪 **Feature engineering**: Incorporate domain-specific knowledge to enhance predictions.
- 📊 **Clustering**: Group wines based on their physicochemical attributes.

---

## 🤝 Contribution
We welcome contributions from the community! Fork the repository, create a new branch, and submit your pull request. Let’s make wine analysis even better together. 🍇

---

## 📜 License
This project is licensed under the **MIT License**. See the LICENSE file for details.

---

### 📫 Contact
Have questions or suggestions? Reach out via [GitHub Issues](https://github.com/your-repo/issues).

Cheers to better wine analysis! 🥂

