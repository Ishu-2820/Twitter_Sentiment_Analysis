# Twitter Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Sentiment%20Analysis-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

A robust machine learning model for classifying airline-related tweets as positive or negative sentiment, providing valuable insights into customer feedback and public opinion.

## 📖 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Pipeline](#technical-pipeline)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a **binary sentiment classification system** using Natural Language Processing (NLP) techniques to analyze airline-related tweets. The model achieves **91% accuracy** in distinguishing between positive and negative customer sentiments, making it valuable for:

- **Customer feedback analysis**
- **Brand reputation monitoring** 
- **Social media sentiment tracking**
- **Business intelligence insights**

### Key Features

- ✅ Text preprocessing with NLTK
- ✅ Feature extraction using CountVectorizer
- ✅ Logistic Regression classification
- ✅ Comprehensive model evaluation
- ✅ Data visualization with matplotlib/seaborn
- ✅ Handles class imbalanced datasets

## 📊 Dataset

The project uses the **Airline Dataset** containing airline-related tweets with the following characteristics:

- **Total tweets**: ~14,000
- **Classes**: Positive (20.5%) and Negative (79.5%)
- **Data source**: Twitter airline customer feedback
- **Format**: CSV file with text and sentiment labels

### Class Distribution
- **Negative tweets**: 79.5% 
- **Positive tweets**: 20.5%

*Note: The dataset exhibits class imbalance, which is addressed in the analysis.*

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (run once)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6
matplotlib>=3.4.0
seaborn>=0.11.0
re
```

## 💻 Usage

### Basic Usage

1. **Prepare your data**
   - Place the `Airline dataset.csv` file in the project directory
   - Ensure the CSV contains 'text' and 'airline_sentiment' columns

2. **Run the analysis**
   ```bash
   python sentiment_analysis.py
   ```

3. **View results**
   - Model metrics will be displayed in the console
   - Visualizations will be saved as image files
   - Confusion matrix and classification report will be generated

### Code Example

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Load and preprocess data
analyzer.load_data('Airline dataset.csv')
analyzer.preprocess_text()

# Train the model
analyzer.train_model()

# Make predictions
sample_text = "I love this airline service!"
prediction = analyzer.predict(sample_text)
print(f"Sentiment: {prediction}")  # Output: Positive
```

## 📁 Project Structure

```
twitter-sentiment-analysis/
│
├── data/
│   └── Airline dataset.csv          # Dataset file
│
├── src/
│   ├── __init__.py
│   ├── sentiment_analyzer.py        # Main analysis script
│   ├── preprocessing.py             # Text preprocessing functions
│   └── visualization.py             # Data visualization functions
│
├── notebooks/
│   └── analysis.ipynb              # Jupyter notebook for exploration
│
├── results/
│   ├── confusion_matrix.png        # Model performance visualization
│   ├── sentiment_distribution.png  # Data distribution charts
│   └── model_metrics.txt          # Detailed performance metrics
│
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
└── LICENSE                        # License file
```

## 🔧 Technical Pipeline

The sentiment analysis follows a comprehensive data science pipeline:

### 1. Data Preparation
- Load airline tweet dataset
- Filter for positive and negative sentiments only
- Handle missing values and duplicates

### 2. Text Preprocessing
```python
# Text cleaning steps:
- Remove URLs and mentions (@user)
- Remove special characters and numbers
- Convert to lowercase
- Remove stop words
- Apply stemming using Porter Stemmer
```

### 3. Feature Engineering
- **CountVectorizer**: Converts text to numerical features
- **Bag of Words**: Creates feature matrix from processed text
- **Vocabulary size**: Optimized for best performance

### 4. Model Training
- **Algorithm**: Logistic Regression
- **Train/Test split**: 80/20
- **Cross-validation**: Implemented for robust evaluation

### 5. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion matrix, ROC curves
- **Analysis**: Class-wise performance breakdown

## 📈 Model Performance

### Overall Metrics
| Metric | Score |
|---------|-------|
| **Accuracy** | 91% |
| **Precision (Weighted)** | 90% |
| **Recall (Weighted)** | 91% |
| **F1-Score (Weighted)** | 90% |

### Class-wise Performance

#### Negative Class
- **Precision**: 93%
- **Recall**: 97%
- **F1-Score**: 95%
- **Support**: 1,836 samples

#### Positive Class  
- **Precision**: 85%
- **Recall**: 70%
- **F1-Score**: 77%
- **Support**: 473 samples

### Confusion Matrix
```
                 Predicted
Actual    Negative  Positive
Negative    1779      57
Positive     143     330
```

**Key Insights:**
- Model excels at identifying negative tweets (97% recall)
- Challenges with positive class due to data imbalance
- False positive rate: 3.1%
- False negative rate: 30.2%

## 📊 Visualizations

The project generates several informative visualizations:

1. **Sentiment Distribution**
   - Pie chart showing class proportions
   - Bar chart of tweet counts by sentiment

2. **Text Analytics**
   - Tweet length distribution histogram
   - Word frequency analysis
   - Most common words in positive vs negative tweets

3. **Model Performance**
   - Confusion matrix heatmap
   - ROC curve and AUC score
   - Precision-Recall curves

4. **Feature Analysis**
   - Most important features for classification
   - Word clouds for each sentiment class

## 🎯 Results

### Key Findings

1. **High Overall Accuracy**: 91% classification accuracy demonstrates strong model performance
2. **Class Imbalance Impact**: Model bias toward negative class due to 79.5% negative tweets
3. **Business Value**: Reliable negative sentiment detection (93% precision) valuable for customer service
4. **Improvement Opportunity**: Positive sentiment detection could benefit from balanced training data

### Business Applications

- **Customer Service**: Automatically flag negative feedback for immediate attention
- **Brand Monitoring**: Track sentiment trends over time
- **Product Insights**: Identify common complaints and praise patterns
- **Marketing Strategy**: Understand customer sentiment drivers

## 🔮 Future Improvements

### Technical Enhancements
- [ ] **Address class imbalance** using SMOTE or weighted sampling
- [ ] **Advanced models**: Experiment with Random Forest, SVM, or Neural Networks
- [ ] **Deep learning**: Implement LSTM or BERT for better context understanding
- [ ] **Feature engineering**: Add n-grams, TF-IDF, or word embeddings
- [ ] **Hyperparameter tuning**: Optimize model parameters using GridSearchCV

### Data & Evaluation
- [ ] **Larger dataset**: Include more balanced positive/negative samples
- [ ] **Multi-class classification**: Extend to neutral sentiment
- [ ] **Cross-domain testing**: Evaluate on other industry tweets
- [ ] **Real-time processing**: Implement streaming sentiment analysis
- [ ] **A/B testing**: Compare different preprocessing approaches

### Production Features
- [ ] **Web API**: Deploy model as REST API
- [ ] **Dashboard**: Create real-time sentiment monitoring dashboard
- [ ] **MLOps**: Implement model versioning and automated retraining
- [ ] **Integration**: Connect with Twitter API for live analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- Model improvements and new algorithms
- Additional visualizations and analysis
- Code optimization and refactoring
- Documentation enhancements
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Airline Twitter Sentiment Dataset
- **Libraries**: scikit-learn, NLTK, pandas, matplotlib, seaborn
- **Inspiration**: Customer feedback analysis and NLP research community
- **Contributors**: Thanks to all contributors who helped improve this project

## 📞 Contact

---

⭐ **If you found this project helpful, please give it a star!** ⭐

*Last updated: August 2025*
