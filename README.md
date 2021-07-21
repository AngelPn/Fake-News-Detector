# Fake-News-Detector
A Fake News detection notebook using classifiers.

# About
[Assignment](https://github.com/AngelPn/Fake-News-Detector/blob/main/2nd-Assignment-2021.pdf) in the course "Data Mining" at DIT - UoA. The notebook detects fake news in news articles. For the sake of simplicity, we say a news article is fake if it has a fake sentiment associated with it. So, the task is to classify fake news from other news. Formally, given a training sample of news and labels, where label ‘1’ denotes non-fake news article and label ‘0’ denotes fake news article, your objective is to predict the labels on the given test dataset. The datasets are loaded from the files into [News-dataset](https://github.com/AngelPn/Fake-News-Detector/tree/main/News_dataset) directory.

## Preprocessing of data
The **NLTK** is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language. **Lemmatization** in NLTK is the algorithmic process of finding the lemma of a word depending on its meaning and context. Lemmatization usually refers to the morphological analysis of words, which aims to remove inflectional endings. It helps in returning the base or dictionary form of a word known as the lemma. For data preprocessing, we write a function which takes each sentence of a corpus as input and returns the lemmatized version of the word. We then use a second method of preprocessing: **Stemming** is a kind of normalization for words. It is a technique where a set of words in a sentence are converted into a sequence to shorten its lookup. The nltk provides many inbuilt stemmers and we will use Snowball Stemmer for that purpose. We write a function for data preprocessing as well, using stemming to accomplish better results.

## Feature Extraction
To analyse the preprocessed data, it needs to be converted into features. We represent words in a numeric format that is understandable by the computers and we use three approached to accomplish that:
1. The **Bag of Words** approach is one of the simplest word embedding approaches. We convert a collection of text documents to a matrix of token counts with CountVectorizer.
2. The **TF-IDF** scheme is a type of bag words approach where instead of adding zeros and ones in the embedding vector, you add floating numbers that contain more useful information compared to zeros and ones. We convert a collection of raw documents to a matrix of TF-IDF features with TfidfVectorizer.
3. Using the word vectors from **Word2Vec** model to create a vector representation for an entire news article (sentence) by taking the mean of all the word vectors present in the news article. 

## Feature-processing
We use feature normalization in machine learning to make model training less sensitive to the scale of features.

## Classifiers
The classifiers that are used:
* **Logistic Regression**: The training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’.
* **Naive Bayes**: Naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features.
* **Support Vector Machine**: Support-vector machines are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.
* **Random Forests**: Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees.
* **Voting Classifier**: A Voting Classifier is a machine learning model that trains on an ensemble of numerous models and predicts an output (class) based on their highest probability of chosen class as the output.

## Evaluation Metrics
We evaluate and record the performance of each method in test data using the following evaluation metrics:
* **Accuracy score**: It is the fraction of predictions our model got right.
* **F1 score**: It is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. It is suitable for uneven class distribution problems.

## Process and conclusions
Ιnitially, we clean our text data using nltk, then we learn about 3 different types of feature-set that we can extract from any text data, and finally we use these feature-sets to build models for sentiment analysis. Below is a summary table showing Accuracy and F1 scores for different models and feature-sets. The conclusions are contained inside the notebook.

# Authors
* [Dora Panteliou](https://github.com/dora-jpg)
* [Angelina Panagopoulou](https://github.com/AngelPn)
