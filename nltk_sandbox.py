import nltk
import random
from nltk.corpus import movie_reviews


def find_features(document, w_features):
    words = set(document)
    features = dict()
    for word in w_features:
        features[word] = (word in words)
    return features


def run_naive_bayes(docs):
    random.shuffle(docs)
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())
    all_words = nltk.FreqDist(all_words)

    word_features = list(all_words.keys())[:3000]

    feature_sets = [(find_features(rev, word_features), category) for (rev, category) in docs]

    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

    clasifier = nltk.NaiveBayesClassifier.train(training_set)
    print(nltk.classify.accuracy(clasifier, testing_set) * 100)
    clasifier.show_most_informative_features(20)


if __name__ == "__main__":
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    run_naive_bayes(documents)




    print("a")