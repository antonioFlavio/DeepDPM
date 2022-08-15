import sys
sys.path.append('./')


import os
import torch
import argparse
import csv
import numpy as np
import re
import umap

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from src import datasets
from tqdm import tqdm

reuters_dataset = '/mnt/d/Datasets/reuters-21578.csv'
the20news_dataset = '/mnt/d/Datasets/the20news.csv'


sentence_transformer = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
le = preprocessing.LabelEncoder()

def split_document(doc):
    sentences = doc.split('.')
    sentences = [s.strip() for s in sentences]
    sentences = list(filter(len, sentences))
    return sentences


def preprocess(text):
    """pre-processing documents before using them for creating the KCG"""
    text = text.lower()
    text_p = "".join([char for char in text if char not in
                      '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~1234567890'])  # removing punctuations except dot (.), and numbers
    words = word_tokenize(text_p)
    stop_words = stopwords.words('english')
    porter = PorterStemmer()

    stemmed_text = ''
    for w in words:
        if w not in stop_words:
            stemmed_text += porter.stem(w) + ' '

    stemmed_text = re.sub(r"[ .]*[.][ ]*", " . ", stemmed_text)
    return stemmed_text[:-1]

def fetch_dataset(dataset_path):
    """

    :param dataset_path: paths.reuters_dataset or paths.the20news_dataset
    :return: numpy array of [label, document]
    """

    with open(dataset_path) as ds:
        reader = csv.reader(ds, delimiter=',')
        headers = next(reader)
        data = []
        for row in reader:
            data.append(row)
    print('dataset is loaded')
    return np.array(data, dtype='object')

def parse_minimal_args(parser):
    # Dataset parameters
    parser.add_argument("--dir", default="/path/to/datasets", help="datasets directory")
    parser.add_argument("--dataset", default="mnist", help="the dataset used")

    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "--ae_pretrain_path", type=str, default="/path/to/ae/weights", help="the path to pretrained ae weights"
    )
    parser.add_argument(
        "--umap_dim", type=int, default=10
    )
    parser.add_argument(
        "--gpus", type=int, default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )    
    return parser

def get_embeddings_conjunto(docs_train):
    codes = []

    for index in tqdm(range(len(docs_train)), "Gerando embeddings"):
        doc = preprocess(docs_train[index])

        documents_sentences = split_document(doc)

        embeddings_list = sentence_transformer.encode(documents_sentences)        
        average_embeddings = np.average(embeddings_list, axis=0)
        #codes.append(torch.from_numpy(average_embeddings))  
        codes.append(average_embeddings)          

    return codes

def make_embbedings():
    parser = argparse.ArgumentParser(description="Only_for_embbedding")
    parser = parse_minimal_args(parser)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus is not None else "cpu")

    data = fetch_dataset(reuters_dataset)
    documents_labels = data[:, 0]
    le.fit(documents_labels)
    documents = data[:, 1]

    n_neighbors = len(set(documents_labels))

    # documents = documents[:200]
    # documents_labels = documents_labels[:200]

    codes = get_embeddings_conjunto(documents)

    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=128)
    print("Starting umap for train...")
    codes = umap_obj.fit_transform(codes)

    labels = le.transform(documents_labels)
    labels = torch.FloatTensor(labels) 

    train_codes, val_codes, train_labels, val_labels = train_test_split(codes, labels)

    # Dados dataset treinamento   

    # train_codes = []
    # train_codes = get_embeddings_conjunto(docs_train)    
    
    # train_labels = le.transform(labels_train)
    # train_labels = torch.FloatTensor(train_labels)    

    torch.save(torch.from_numpy(train_codes), f'train_codes_.pt')
    torch.save(train_labels, f'train_labels.pt')
    
    # Dados dataset validação
    
    # val_codes = get_embeddings_conjunto(docs_val)        

    # val_labels = le.transform(labels_val)
    # val_labels = torch.FloatTensor(val_labels)    

    # val_codes = torch.cat(val_codes)    

    torch.save(torch.from_numpy(val_codes), f'val_codes.pt')
    torch.save(val_labels, f'val_labels.pt')

if __name__ == "__main__":
    make_embbedings()