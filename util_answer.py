import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import *
def load_words(filepath: str):
    """
    read file and load parsed words with jieba, each line is one document
    Args:
        filepath: file path to load
    Returns:
        terms: List[str], a list of all words
        documents: List[List[str]], a list of documents, each sublist contains words in the document
    """
    f = open(filepath)
    documents = []
    terms = []
    for line in f.readlines():
        words = jieba.lcut(line.strip(), cut_all=False)
        doc_words = []
        for word in words:
            for char in word:
                num = ord(char)
                if not ((num >= 0x4E00 and num <= 0x9FFF) or (num >= 0x3400 and num <= 0x4DBF) or (num >= 0x20000 and num <= 0x2A6DF) or (num >= 0x2A700 and num <= 0x2B73F) or (num >= 0x2B740 and num <= 0x2B81F) or (num >= 0x2B820 and num <= 0x2CEAF) or (num >= 0xF900 and num <= 0xFAFF) or (num >= 0x2F800 and num <= 0x2FA1F)):
                    continue
                doc_words.append(word)
        if len(doc_words) > 0:
            documents.append(doc_words)
            terms.extend(doc_words)
    f.close()
    return terms, documents

def build_term_doc_matrix(documents: List[List[str]], terms: List[str]):
    """
    build term-document matrix
    Args:
        documents: List[List[str]], a list of documents, each sublist contains words in the document, len(documents) = N
        terms: List[str], a list of all words, len(terms) = D
    Returns:
        term_doc: np.array, of size N * D, where term_doc[i, j] means the number of times word_j appears in document_i
    """
    term_doc = []
    for term in terms:
        tmp = []
        for document in documents:
            tmp.append(document.count(term))
        term_doc.append(tmp)
    term_doc = np.array(term_doc)
    return term_doc

def cal_tfidf_matrix(term_doc: np.array, documents: List[List[str]], terms: List[str]):
    """
    calculate TF-IDF value for each word
    Args:
        term_doc: N * D term-document matrix
        documents: List[List[str]], a list of documents, each sublist contains words in the document, len(documents) = N
        terms: List[str], a list of all words, len(terms) = D
    Returns:
        TF-IDF: Dict[int, float], where TF-IDF[word_i] is the tf-odf value for word_i
    """
    TF_IDF = {}
    TF = term_doc.sum(axis=1) / term_doc.sum()
    IDF = np.log(len(documents) / (term_doc.astype(bool).sum(axis=1) + 1))
    for i in range(len(terms)):
        TF_IDF[terms[i]] = TF[i] * IDF[i]
    return TF_IDF

def search_key_similarity(U: np.array, s: np.array, VT: np.array, terms: List[str], term_doc: np.array, keys: List[str], k: int=10):
    """
    calculate cosine similarity for each pair of key and document
    U, s, VT are SVD results of term-document matrix
    ```
    U, s, VT = scipy.linalg.svd(term_doc)
    ```
    Args:
        U: numpy matrix of size (D,D)
        s: numpy array of size (N,)
        VT: numpy array of size 
        terms: List[str], a list of all words, len(terms) = D
        term_doc: N * D term-document matrix
        keys: a list of search keys
        k: number of features used in matrix approximation
    Returns:
        sim_matrix: np.array of size (N, len(keys))
    """
    Sigma = np.zeros((term_doc.shape[0], term_doc.shape[1]))
    Sigma[:term_doc.shape[1], :term_doc.shape[1]] = np.diag(1/s)
    _U = U[:, :k]
    _Sigma = Sigma[:k, :k]
    _VT = VT[:k, :]
    query = _U.dot(_Sigma)
    my = []
    for key in keys:
        temp = jieba.lcut(key.strip(), cut_all=False)
        q = []
        for j in terms:
            if j in temp:
                q.append(1)
            else:
                q.append(0)
        my.append(q)
    my = np.array(my)
    sim_matrix = cosine_similarity(_VT.T, my.dot(query))

    return sim_matrix

def classification(sim_matrix: np.array):
    """
    view as a document classification problem, return the most similar key index for each document
    Args:
        sim_matrix: np.array of size (N, len(keys))
    Returns:
        predict: np.array of size (N,)
    """
    predict = np.argmax(abs(sim_matrix), axis=1)
    return predict

def search_topn_for_each_key(sim_matrix: np.array, n: int=10):
    """
    view as a search problem, return the top-n most similar document index for each keyword
    Args:
        sim_matrix: np.array of size (N, len(keys))
    Returns:
        searched: np.array of size (len(keys), n)
    """
    searched = np.argsort(abs(sim_matrix), axis=0)[-n:].T
    return searched