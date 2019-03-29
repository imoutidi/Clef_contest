import json
import sklearn.datasets as skd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np


def multi_nb():
    categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
    news_train = skd.load_files(container_path="/home/iraklis/PycharmProjects/Clef_contest/I_O/20 Newsgroup dataset/"
                                               "20news-bydate-train",
                                categories=categories, encoding="ISO-8859-1")
    news_test = skd.load_files(container_path="/home/iraklis/PycharmProjects/Clef_contest/I_O/20 Newsgroup dataset/"
                                              "20news-bydate-test",
                               categories=categories, encoding="ISO-8859-1")

    text_clf = Pipeline([('vect', TfidfVectorizer()),
                         ('clf', MultinomialNB())])

    # training
    text_clf.fit(news_train.data, news_train.target)
    # predicting
    predictions = text_clf.predict(news_test.data)

    print('Accuracy achieved is ' + str(np.mean(predictions == news_test.target)))
    print(metrics.classification_report(news_test.target, predictions, target_names=news_test.target_names)),
    metrics.confusion_matrix(news_test.target, predictions)


def json_parsing():
    with open("/home/iraklis/PycharmProjects/Clef_contest/I_O/output/Document/"
              "dev_filled.json", "r") as json_data:
        count = 0
        for line in json_data:
            d = json.loads(line)
            count += 1
            # print("a")
        print(count)


if __name__ == "__main__":
    multi_nb()
    print("a")
