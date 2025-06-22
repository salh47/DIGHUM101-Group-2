Program Florian
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from string import punctuation
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# %%
import pandas as pd
import os
data = pd.read_csv('Data/us-nara-amending-america-dataset-raw-2016-02-25 (1).csv', encoding='ISO-8859-1')
data.head()

# %%
data = data["title_or_description_from_source"]
data.head()

# %%
def preprocess_text_list(data):
    processed_list = []
    for text in data:
        # Tokenize into sentences
        tokens = word_tokenize(str(text).lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        processed_list.append(" ".join(tokens))
    return processed_list
processed_texts = preprocess_text_list(data)
print(processed_texts[:])

# %%
wordsplit = [word for processed_texts in processed_texts for word in processed_texts.split()]
print(wordsplit)

# %%
from collections import Counter
freq2 = Counter(wordsplit)
freq2.most_common(50)

# %%
pip install spacy

# %%
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"]) 

# %%
doc = nlp(" ".join(wordsplit))

# %%
import spacy
nlp = spacy.load("en_core_web_sm")

# %%
lemmas = [token.lemma_ for token in doc]
print(lemmas)

# %%
from collections import Counter
freq3 = Counter(lemmas)
freq3.most_common(50)

# %%
freq3 = pd.DataFrame(freq3.most_common(50), columns=['word', 'frequency'])
freq3

# %%
data = pd.read_csv('Data/us-nara-amending-america-dataset-raw-2016-02-25 (1).csv', encoding='ISO-8859-1')

def most_common_year_from_df(word):
    # Filter rows where the word appears in the text
    mask = data['title_or_description_from_source'].str.lower().str.contains(r'\b{}\b'.format(word), na=False)
    years = data.loc[mask, 'year']
    if not years.empty:
        return years.value_counts().idxmax()
    return None

freq3["Year mostly appear"] = freq3["word"].apply(most_common_year_from_df)
freq3

# %%
number_documents = len(data)
print(number_documents)

# %%
from nltk import word_tokenize # why this and not split()?

number_of_documents_with_document = sum(1 for doc in processed_texts if 'president' in word_tokenize(doc))
print("Number of documents with the word president:", number_of_documents_with_document)

# %%
idf = number_documents/number_of_documents_with_document
print("Inverse Document Frequency for 'president':", idf)

# %%
tf = [word_tokenize(doc).count('president') for doc in processed_texts]
print(tf)

# %%
tf_idf = []
for i, doc in enumerate(processed_texts):
    tf_i = word_tokenize(doc).count('president')
    doc_length = len(word_tokenize(doc))
    if doc_length > 0:
        tf_normalized = tf_i / doc_length
    else:
        tf_normalized = 0
    tfidf_value = tf_normalized * idf
    tf_idf.append(tfidf_value)
    print(f"President {i}: TF = {tf_i}, Doc Length = {doc_length}, TF_norm = {tf_normalized:.2f}, IDF = {idf:.2f}, TF-IDF = {tfidf_value:.4f}")

# %%
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
stop = stopwords.words('english')

tfidfvec = TfidfVectorizer(min_df=2,lowercase=True,stop_words=stop)
# min_df=2 means we only consider words that appear in at least 2 documents

tfidf_bow = tfidfvec.fit_transform(processed_texts)
print(tfidf_bow.toarray())
print(tfidf_bow.shape)

# %%
#It worked without using the ', index=processed_texts.index' at the end.
tfidf = pd.DataFrame(tfidf_bow.toarray(), columns=tfidfvec.get_feature_names_out())
tfidf.head(5)

# %%
tfidf.max().sort_values(ascending=False).head(50)

# %%
print(type(wordsplit))
text_set = set(wordsplit)
print(text_set)


