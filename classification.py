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

from neural_network import NeuralCreator


class ClassificationTask:

    def __init__(self):
        self.train_txt = pandas.Series()
        self.test_txt = pandas.Series()
        self.train_lbl = pandas.Series()
        self.test_lbl = pandas.Series()
        self.data_df = pandas.DataFrame()
        self.xtrain_count = 0
        self.xtest_count = 0
        self.xtrain_tfidf = 0
        self.xtest_tfidf = 0
        self.xtrain_tfidf_ngram = 0
        self.xtest_tfidf_ngram = 0
        self.xtrain_tfidf_ngram_chars = 0
        self.xtest_tfidf_ngram_chars = 0
        self.embedding_matrix = numpy.zeros(0)
        self.word_index = {}
        self.train_seq_x = 0
        self.test_seq_x = 0

    # preprocessing the CLEF dataset
    def json_parsing(self):
        encoder = preprocessing.LabelEncoder()
        train_text = list()
        train_labels = list()
        with open("/home/iraklis/PycharmProjects/Clef_contest/I_O/data/"
                  "clef_data/Document/train_filled.json", "r") as json_data:
            for line in json_data:
                json_doc = json.loads(line)
                train_text.append(json_doc['text'])
                train_labels.append(json_doc['label'])
        self.train_txt = pandas.Series(train_text)
        self.train_lbl = encoder.fit_transform(pandas.Series(train_labels))

        test_text = list()
        test_labels = list()
        with open("/home/iraklis/PycharmProjects/Clef_contest/I_O/data/"
                  "clef_data/Document/dev_filled.json", "r") as json_data:
            for line in json_data:
                json_doc = json.loads(line)
                test_text.append(json_doc['text'])
                test_labels.append(json_doc['label'])
        self.test_txt = pandas.Series(test_text)
        self.test_lbl = encoder.fit_transform(pandas.Series(test_labels))

        all_text = train_text + test_text
        all_labels = train_labels + test_labels
        self.data_df['text'] = all_text
        self.data_df['label'] = all_labels

    # preprocessing some example datasets
    def pre_process(self, data_path):
        data = open(data_path).read()
        labels, texts = [], []
        # for i, line in enumerate(data.split("\n")):
        #     content = line.split()
        #     labels.append(content[0])
        #     texts.append(" ".join(content[1:]))
        #
        # # create a dataframe using texts and lables
        # self.train_df['text'] = texts
        # self.train_df['label'] = labels
        #
        # # split the dataset into random train and test subsets
        # self.train_x, self.test_x, self.train_y, self.test_y = \
        #     model_selection.train_test_split(self.train_df['text'], self.train_df['label'])
        #
        # # label encode the target variable
        # encoder = preprocessing.LabelEncoder()
        # self.train_y = encoder.fit_transform(self.train_y)
        # self.test_y = encoder.fit_transform(self.test_y)

    def vectorize(self):
        # create a count vectorizer object
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(self.data_df['text'])
        self.xtrain_count = count_vect.transform(self.train_txt)
        self.xtest_count = count_vect.transform(self.test_txt)

        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(self.data_df['text'])
        self.xtrain_tfidf = tfidf_vect.transform(self.train_txt)
        self.xtest_tfidf = tfidf_vect.transform(self.test_txt)

        # word level ngram tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                           max_features=5000)
        tfidf_vect_ngram.fit(self.data_df['text'])
        self.xtrain_tfidf_ngram = tfidf_vect_ngram.transform(self.train_txt)
        self.xtest_tfidf_ngram = tfidf_vect_ngram.transform(self.test_txt)

        # character level ngram tf-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                                 max_features=5000)
        tfidf_vect_ngram_chars.fit(self.data_df['text'])
        self.xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(self.train_txt)
        self.xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(self.test_txt)

    def embeding(self, vector_path):
        # load the pre-trained word-embedding vectors
        embeddings_index = {}
        for i, line in enumerate(open(vector_path)):
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

        # create a tokenizer
        token = text.Tokenizer()
        token.fit_on_texts(self.data_df['text'])
        self.word_index = token.word_index

        # convert text to sequence of tokens and pad them to ensure equal length vectors
        self.train_seq_x = sequence.pad_sequences(token.texts_to_sequences(self.train_txt), maxlen=70)
        self.test_seq_x = sequence.pad_sequences(token.texts_to_sequences(self.test_txt), maxlen=70)

        # create token-embedding mapping
        self.embedding_matrix = numpy.zeros((len(self.word_index) + 1, 300))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        return metrics.accuracy_score(predictions, self.test_lbl)

    def run_naive_bayes(self):
        print("Running naive bayes.")
        # Naive Bayes on Count Vectors
        accuracy = self.train_model(naive_bayes.MultinomialNB(), self.xtrain_count,
                                    self.train_lbl, self.xtest_count)
        print("NB, Count Vectors: ", accuracy)

        # Naive Bayes on Word Level TF-IDF Vectors
        accuracy = self.train_model(naive_bayes.MultinomialNB(), self.xtrain_tfidf,
                                    self.train_lbl, self.xtest_tfidf)
        print("NB, WordLevel TF-IDF: ", accuracy)

        # Naive Bayes on Ngram Level TF-IDF Vectors
        accuracy = self.train_model(naive_bayes.MultinomialNB(), self.xtrain_tfidf_ngram,
                                    self.train_lbl, self.xtest_tfidf_ngram)
        print("NB, N-Gram Vectors: ", accuracy)

        # Naive Bayes on Character Level TF-IDF Vectors
        accuracy = self.train_model(naive_bayes.MultinomialNB(), self.xtrain_tfidf_ngram_chars,
                                    self.train_lbl, self.xtest_tfidf_ngram_chars)
        print("NB, CharLevel Vectors: " + str(accuracy) + "\n")

    def run_logistic_regression(self):
        print("Running logistic regression.")
        # Linear Classifier on Count Vectors
        accuracy = self.train_model(linear_model.LogisticRegression(), self.xtrain_count,
                                    self.train_lbl, self.xtest_count)
        print("LR, Count Vectors: ", accuracy)

        # Linear Classifier on Word Level TF-IDF Vectors
        accuracy = self.train_model(linear_model.LogisticRegression(), self.xtrain_tfidf,
                                    self.train_lbl, self.xtest_tfidf)
        print("LR, Word Level TF-IDF: ", accuracy)

        # Linear Classifier on Ngram Level TF-IDF Vectors
        accuracy = self.train_model(linear_model.LogisticRegression(), self.xtrain_tfidf_ngram,
                                    self.train_lbl, self.xtest_tfidf_ngram)
        print("LR, N-Gram Vectors: ", accuracy)

        # Linear Classifier on Character Level TF-IDF Vectors
        accuracy = self.train_model(linear_model.LogisticRegression(), self.xtrain_tfidf_ngram_chars,
                                    self.train_lbl, self.xtest_tfidf_ngram_chars)
        print("LR, CharLevel Vectors: " + str(accuracy) + "\n")

    def run_svm(self):
        print("Running SVM.")
        # SVM on word Level TF-IDF Vectors
        accuracy = self.train_model(svm.SVC(), self.xtrain_tfidf, self.train_lbl,
                                    self.xtest_tfidf)
        print("SVM, Word Level TF-IDF: ", accuracy)
        # SVM on Ngram Level TF-IDF Vectors
        accuracy = self.train_model(svm.SVC(gamma="auto"), self.xtrain_tfidf_ngram, self.train_lbl, self.xtest_tfidf_ngram)
        print("SVM, N-Gram Vectors: " + str(accuracy) + "\n")

    def run_random_forest(self):
        print("Running random forest.")
        # Random Forest on Count Vectors
        accuracy = self.train_model(ensemble.RandomForestClassifier(), self.xtrain_count, self.train_lbl, self.xtest_count)
        print("Random Forest, Count Vectors: ", accuracy)

        # Random Forest on Word Level TF IDF Vectors
        accuracy = self.train_model(ensemble.RandomForestClassifier(), self.xtrain_tfidf, self.train_lbl, self.xtest_tfidf)
        print("Random Forest, WordLevel TF-IDF: " + str(accuracy) + "\n")

    def run_xgb(self):
        print("Running extreme gradient boosting.")
        # Extereme Gradient Boosting on Count Vectors
        accuracy = self.train_model(xgboost.XGBClassifier(), self.xtrain_count.tocsc(), self.train_lbl,
                                    self.xtest_count.tocsc())
        print("Xgb, Count Vectors: ", accuracy)

        # Extereme Gradient Boosting on Word Level TF IDF Vectors
        accuracy = self.train_model(xgboost.XGBClassifier(), self.xtrain_tfidf.tocsc(), self.train_lbl,
                                    self.xtest_tfidf.tocsc())
        print("Xgb, WordLevel TF-IDF: ", accuracy)

        # Extereme Gradient Boosting on Character Level TF IDF Vectors
        accuracy = self.train_model(xgboost.XGBClassifier(), self.xtrain_tfidf_ngram_chars.tocsc(), self.train_lbl,
                               self.xtest_tfidf_ngram_chars.tocsc())
        print("Xgb, CharLevel Vectors: " + str(accuracy) + "\n")

    def run_neural(self):
        neural_net = NeuralCreator(self.xtrain_tfidf_ngram.shape[1])
        shallow_classifier = neural_net.create_shallow_model()
        accuracy = self.train_model(shallow_classifier, self.xtrain_tfidf_ngram, self.train_lbl,
                                    self.xtest_tfidf_ngram, is_neural_net=True)
        print("NN, Ngram Level TF IDF Vectors" + str(accuracy))

        cnn_classifier = neural_net.create_cnn(self.word_index, self.embedding_matrix)
        accuracy = self.train_model(cnn_classifier, self.train_seq_x, self.train_lbl,
                                    self.test_seq_x, is_neural_net=True)
        print("CNN, Word Embeddings" + str(accuracy))

        rnn_gru_classifier = neural_net.create_rnn_gru(self.word_index, self.embedding_matrix)
        accuracy = self.train_model(rnn_gru_classifier, self.train_seq_x, self.train_lbl,
                                    self.test_seq_x, is_neural_net=True)
        print("RNN-GRU, Word Embeddings" + str(accuracy))

        bi_rnn_classifier = neural_net.create_bidirectional_rnn(self.word_index, self.embedding_matrix)
        accuracy = self.train_model(bi_rnn_classifier, self.train_seq_x, self.train_lbl,
                                    self.test_seq_x, is_neural_net=True)
        print("RNN-Bidirectional, Word Embeddings" + str(accuracy))

        rcnn_classifier = neural_net.create_rcnn(self.word_index, self.embedding_matrix)
        accuracy = self.train_model(rcnn_classifier, self.train_seq_x, self.train_lbl,
                                    self.test_seq_x, is_neural_net=True)
        print("CNN, Word Embeddings", str(accuracy))


if __name__ == "__main__":

    c_task = ClassificationTask()
    c_task.json_parsing()
    # c_task.pre_process('I_O/data/corpus')
    c_task.vectorize()
    c_task.embeding('I_O/data/wiki-news-300d-1M.vec')
    c_task.run_naive_bayes()
    c_task.run_logistic_regression()
    c_task.run_svm()
    c_task.run_random_forest()
    c_task.run_xgb()
    c_task.run_neural()
    print("a")
