import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize_stopwords_stemmer(text, stemmer):
    no_punctuation = text.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    text_filter = [w for w in tokens if not w in stopwords.words('english')]
    text_final = []
    for k in text_filter:
        text_final.append(stemmer.stem(k))
    return text_final

def organizes_documents():
	files = open('C:\Users\Alex Souza\Google Drive\3. Estudos e Projetos\2. MESTRADO\UECE\Regular\TÓPICOS EM INTELIGENCIA COMPUTACIONAL\Bases\med', 'r').read().split('.I')
	stemmer = PorterStemmer()
	for i in range(1,100):
		text = files[i].replace('.W', '')
		text = text.replace(str(i), '')
		text_trans['Doc-'+str(i)] = tokenize_stopwords_stemmer(text.lower(), stemmer)
		pass	
text_trans = {}

organizes_documents()
tfidf = TfidfVectorizer(lowercase=False)
v = text_trans.values()
matrix_tfidf = tfidf.fit_transform(v[0])
