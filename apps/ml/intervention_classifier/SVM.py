# for doc2vec modelling
from nltk.stem import WordNetLemmatizer
from multiprocessing import cpu_count, Pool  # for multiprocessing data
from IPython.display import Image as im
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')

stopwords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

cores = cpu_count()


class SVM:
    def __init__(self):
        self.doc2vec_model = Doc2Vec.load(
            'apps/ml/intervention_classifier/models/doc2vec.model')
        self.svm_model = pickle.load(
            open('apps/ml/intervention_classifier/models/svm-model', 'rb'))

    def predict(self, input_data):
        def clean_text(df):
            df['cleaned_text'] = df['narrative'].fillna('')
            df['cleaned_text'] = df['cleaned_text'].str.lower()
            df['cleaned_text'] = df['cleaned_text'].str.replace(
                r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|rt|\d+', '')
            df['cleaned_text'] = df['cleaned_text'].str.replace(
                r'^\s+|\s+$', '')
            df['cleaned_text'] = df['cleaned_text'].apply(
                lambda x: ' '.join([w for w in x.split() if w not in (stopwords)]))
            df['cleaned_split'] = df['cleaned_text'].apply(lambda x: x.split())
            return df

        def lemmatize_df(df):
            df['lemmatized'] = df['cleaned_split'].apply(
                lambda x: [lemmatizer.lemmatize(word) for word in x])
            return df

        df = pd.DataFrame([input_data["message"]], columns=[
                          "narrative"], index=None)

        df = clean_text(df)
        df = lemmatize_df(df)

        # doc2vec
        vectors = self.doc2vec_model.infer_vector(np.array(df.lemmatized)[0])
        prediction = self.svm_model.predict(vectors.reshape(1, -1))

        return prediction[0]

    # Todo
    # Add other metrics and network status details
