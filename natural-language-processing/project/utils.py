import nltk
import pickle
import re
import numpy as np
import gensim

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'CHITCHAT_BOT': 'chitchat_bot.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv'
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path, bin=False):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype
    dim = False
    if bin:
        wv_embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
        dim = len(wv_embeddings[wv_embeddings.keys()[0]])
    else:
        wv_embeddings = {}
        for line in open(embeddings_path):
            w, *r = line.strip().split('\t')
            wv_embeddings[w] = np.array([float(ri) for ri in r], dtype=np.float32)
            if not dim:
                dim = len(wv_embeddings[w])
    
    return wv_embeddings, dim

def question_to_vec(question, embeddings, dim):
    """
        Transforms a string to an embedding by averaging word embeddings.
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    rep = [0]*dim
    words = question.split()
    n = 0
    for w in words:
        if(w in embeddings):
            n+=1
            rep += embeddings[w][:dim]
    if n!= 0:        
        rep[:] = [r/n for r in rep]
    return rep


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
