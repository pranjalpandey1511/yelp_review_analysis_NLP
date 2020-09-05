import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer




def exploratory_data_analysis(yelp_df):
    yelp_df.info()
    yelp_df.head(10)
    yelp_df.tail()
    yelp_df.describe()

    # Length of the messages
    yelp_df['length'] = yelp_df['text'].apply(len)
    yelp_df.head(5)
    yelp_df["length"].describe()
    yelp_df['length'].plot(bins=100, kind='hist')
    plt.show()
    print("The Longest Message",yelp_df[yelp_df['length'] == 4997]['text'].iloc[0])
    print("The shortest Message",yelp_df[yelp_df['length'] == 710]['text'].iloc[0])
    g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=3)
    g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=5)
    g.map(plt.hist, 'length', bins=20, color='r')
    yelp_df_1 = yelp_df[yelp_df['stars'] == 1]
    yelp_df_5 = yelp_df[yelp_df['stars'] == 5]
    yelp_df_1_5 = pd.concat([yelp_df_1, yelp_df_5])
    yelp_df_1_5.info()
    print('1-Stars percentage =', (len(yelp_df_1) / len(yelp_df_1_5)) * 100, "%")
    print('5-Stars percentage =', (len(yelp_df_5) / len(yelp_df_1_5)) * 100, "%")
    sns.countplot(yelp_df_1_5['stars'], label="Count")
    return yelp_df_1_5



def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

def data_format_train(yelp_df_1_5):
    yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)
    print(yelp_df_clean[0])
    print(yelp_df_1_5['text'][0])  # show the original version
    vectorizer = CountVectorizer(analyzer=message_cleaning)
    yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])
    print(vectorizer.get_feature_names())
    print(yelp_countvectorizer.toarray())
    print(yelp_countvectorizer.shape)
    NB_classifier = MultinomialNB()
    label = yelp_df_1_5['stars'].values
    print(label)
    NB_classifier.fit(yelp_countvectorizer, label)
    testing_sample = ['Best food!']
    testing_sample_countvectorizer = vectorizer.transform(testing_sample)
    test_predict = NB_classifier.predict(testing_sample_countvectorizer)
    print(test_predict)
    testing_sample = ['sick, poor food']
    testing_sample_countvectorizer = vectorizer.transform(testing_sample)
    test_predict = NB_classifier.predict(testing_sample_countvectorizer)
    print(test_predict)
    X = yelp_countvectorizer
    y = label
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    NB_classifier = MultinomialNB()
    NB_classifier.fit(X_train, y_train)
    return  X_train, X_test, y_train, y_test, NB_classifier, yelp_countvectorizer, label

def evaulate_Model(X_train, X_test, y_train, y_test, NB_classifier, yelp_countvectorizer, label):
    y_predict_train = NB_classifier.predict(X_train)
    print(y_predict_train)
    cm = confusion_matrix(y_train, y_predict_train)
    sns.heatmap(cm, annot=True)
    plt.show()

    # Predicting the Test set results
    y_predict_test = NB_classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_predict_test)
    sns.heatmap(cm, annot=True)
    plt.show()
    print(classification_report(y_test, y_predict_test))

    #using tfidf
    yelp_tfidf = TfidfTransformer().fit_transform(yelp_countvectorizer)
    print(yelp_tfidf.shape)
    print(yelp_tfidf[:, :])
    X = yelp_tfidf
    y = label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    NB_classifier = MultinomialNB()
    NB_classifier.fit(X_train, y_train)

    y_predict_train = NB_classifier.predict(X_train)
    print(y_predict_train)
    cm = confusion_matrix(y_train, y_predict_train)
    sns.heatmap(cm, annot=True)
    plt.show()

if __name__ == "__main__":
    yelp_df = pd.read_csv("yelpReview.csv")
    yelp_df_cleaned = exploratory_data_analysis(yelp_df)
    X_train, X_test, y_train, y_test, NB_classifier, yelp_countvectorizer, label = data_format_train(yelp_df_cleaned)
    evaulate_Model(X_train, X_test, y_train, y_test, NB_classifier, yelp_countvectorizer, label)

