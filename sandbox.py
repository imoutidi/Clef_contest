import json
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas
import xgboost
import numpy
import textblob
import string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# Ignoring warnings from Xgb classifier
# This will be fixed by sklearn future release
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


def json_parsing():
    with open("/home/iraklis/PycharmProjects/Clef_contest/I_O/output/Document/"
              "dev_filled.json", "r") as json_data:
        count = 0
        for line in json_data:
            d = json.loads(line)
            count += 1
            # print("a")
        print(count)


class ClassificationTask:

    def __init__(self):
        self.train_x = 0
        self.test_x = 0
        self.train_y = 0
        self.test_y = 0
        self.train_df = pandas.DataFrame()
        self.xtrain_count = 0
        self.xtest_count = 0
        self.xtrain_tfidf = 0
        self.xtest_tfidf = 0
        self.xtrain_tfidf_ngram = 0
        self.xtest_tfidf_ngram = 0
        self.xtrain_tfidf_ngram_chars = 0
        self.xtest_tfidf_ngram_chars = 0

    def pre_process(self, data_path):
        data = open(data_path).read()
        labels, texts = [], []
        for i, line in enumerate(data.split("\n")):
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))

        # create a dataframe using texts and lables
        self.train_df['text'] = texts
        self.train_df['label'] = labels

        # split the dataset into random train and test subsets
        self.train_x, self.test_x, self.train_y, self.test_y = \
            model_selection.train_test_split(self.train_df['text'], self.train_df['label'])

        # label encode the target variable
        encoder = preprocessing.LabelEncoder()
        self.train_y = encoder.fit_transform(self.train_y)
        self.test_y = encoder.fit_transform(self.test_y)

    def vectorize(self):
        # create a count vectorizer object
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(self.train_df['text'])
        self.xtrain_count = count_vect.transform(self.train_x)
        self.xtest_count = count_vect.transform(self.test_x)

        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(self.train_df['text'])
        self.xtrain_tfidf = tfidf_vect.transform(self.train_x)
        self.xtest_tfidf = tfidf_vect.transform(self.test_x)

        # word level ngram tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                           max_features=5000)
        tfidf_vect_ngram.fit(self.train_df['text'])
        self.xtrain_tfidf_ngram = tfidf_vect_ngram.transform(self.train_x)
        self.xtest_tfidf_ngram = tfidf_vect_ngram.transform(self.test_x)

        # character level ngram tf-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                                 max_features=5000)
        tfidf_vect_ngram_chars.fit(self.train_df['text'])
        self.xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(self.train_x)
        self.xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(self.test_x)

    def embeding(self, vector_path):
        # load the pre-trained word-embedding vectors
        embeddings_index = {}
        for i, line in enumerate(open(vector_path)):
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

        # create a tokenizer
        token = text.Tokenizer()
        token.fit_on_texts(self.train_df['text'])
        word_index = token.word_index

        # convert text to sequence of tokens and pad them to ensure equal length vectors
        train_seq_x = sequence.pad_sequences(token.texts_to_sequences(self.train_x), maxlen=70)
        test_seq_x = sequence.pad_sequences(token.texts_to_sequences(self.test_x), maxlen=70)

        # create token-embedding mapping
        embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        return metrics.accuracy_score(predictions, self.test_y)

    def run_naive_bayes(self):
        print("Running naive bayes.")
        # Naive Bayes on Count Vectors
        accuracy = self.train_model(naive_bayes.MultinomialNB(), self.xtrain_count,
                                    self.train_y, self.xtest_count)
        print("NB, Count Vectors: ", accuracy)

        # Naive Bayes on Word Level TF-IDF Vectors
        accuracy = self.train_model(naive_bayes.MultinomialNB(), self.xtrain_tfidf,
                                    self.train_y, self.xtest_tfidf)
        print("NB, WordLevel TF-IDF: ", accuracy)

        # Naive Bayes on Ngram Level TF-IDF Vectors
        accuracy = self.train_model(naive_bayes.MultinomialNB(), self.xtrain_tfidf_ngram,
                                    self.train_y, self.xtest_tfidf_ngram)
        print("NB, N-Gram Vectors: ", accuracy)

        # Naive Bayes on Character Level TF-IDF Vectors
        accuracy = self.train_model(naive_bayes.MultinomialNB(), self.xtrain_tfidf_ngram_chars,
                                    self.train_y, self.xtest_tfidf_ngram_chars)
        print("NB, CharLevel Vectors: " + str(accuracy) + "\n")

    def run_logistic_regression(self):
        print("Running logistic regression.")
        # Linear Classifier on Count Vectors
        accuracy = self.train_model(linear_model.LogisticRegression(), self.xtrain_count,
                                    self.train_y, self.xtest_count)
        print("LR, Count Vectors: ", accuracy)

        # Linear Classifier on Word Level TF-IDF Vectors
        accuracy = self.train_model(linear_model.LogisticRegression(), self.xtrain_tfidf,
                                    self.train_y, self.xtest_tfidf)
        print("LR, Word Level TF-IDF: ", accuracy)

        # Linear Classifier on Ngram Level TF-IDF Vectors
        accuracy = self.train_model(linear_model.LogisticRegression(), self.xtrain_tfidf_ngram,
                                    self.train_y, self.xtest_tfidf_ngram)
        print("LR, N-Gram Vectors: ", accuracy)

        # Linear Classifier on Character Level TF-IDF Vectors
        accuracy = self.train_model(linear_model.LogisticRegression(), self.xtrain_tfidf_ngram_chars,
                                    self.train_y, self.xtest_tfidf_ngram_chars)
        print("LR, CharLevel Vectors: " + str(accuracy) + "\n")

    def run_svm(self):
        print("Running SVM.")
        # SVM on word Level TF-IDF Vectors
        accuracy = self.train_model(svm.SVC(), self.xtrain_tfidf, self.train_y,
                                    self.xtest_tfidf)
        print("SVM, Word Level TF-IDF: ", accuracy)
        # SVM on Ngram Level TF-IDF Vectors
        accuracy = self.train_model(svm.SVC(gamma="auto"), self.xtrain_tfidf_ngram, self.train_y, self.xtest_tfidf_ngram)
        print("SVM, N-Gram Vectors: " + str(accuracy) + "\n")

    def run_random_forest(self):
        print("Running random forest.")
        # Random Forest on Count Vectors
        accuracy = self.train_model(ensemble.RandomForestClassifier(), self.xtrain_count, self.train_y, self.xtest_count)
        print("Random Forest, Count Vectors: ", accuracy)

        # Random Forest on Word Level TF IDF Vectors
        accuracy = self.train_model(ensemble.RandomForestClassifier(), self.xtrain_tfidf, self.train_y, self.xtest_tfidf)
        print("Random Forest, WordLevel TF-IDF: " + str(accuracy) + "\n")

    def run_xgb(self):
        print("Running extreme gradient boosting.")
        # Extereme Gradient Boosting on Count Vectors
        accuracy = self.train_model(xgboost.XGBClassifier(), self.xtrain_count.tocsc(), self.train_y,
                                    self.xtest_count.tocsc())
        print("Xgb, Count Vectors: ", accuracy)

        # Extereme Gradient Boosting on Word Level TF IDF Vectors
        accuracy = self.train_model(xgboost.XGBClassifier(), self.xtrain_tfidf.tocsc(), self.train_y,
                                    self.xtest_tfidf.tocsc())
        print("Xgb, WordLevel TF-IDF: ", accuracy)

        # Extereme Gradient Boosting on Character Level TF IDF Vectors
        accuracy = self.train_model(xgboost.XGBClassifier(), self.xtrain_tfidf_ngram_chars.tocsc(), self.train_y,
                               self.xtest_tfidf_ngram_chars.tocsc())
        print("Xgb, CharLevel Vectors: " + str(accuracy) + "\n")


if __name__ == "__main__":
    c_task = ClassificationTask()
    c_task.pre_process('I_O/data/corpus')
    c_task.vectorize()
    # c_task.embeding('I_O/data/wiki-news-300d-1M.vec')
    c_task.run_naive_bayes()
    c_task.run_logistic_regression()
    # c_task.run_svm()
    c_task.run_random_forest()
    c_task.run_xgb()
    print("a")
