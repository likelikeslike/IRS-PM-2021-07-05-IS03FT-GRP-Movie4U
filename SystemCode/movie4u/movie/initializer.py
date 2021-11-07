import numpy as np

from .search_cache import LRU
from .search_index import Index
from .movie_dict import MovieDict
from data.process import Tokenizer
from model.bert_recommender import BertRecommenderConfig, BertRecommender

search_cache = LRU()
search_index = Index()
movie_dicts = MovieDict(
    './data/dict/movie_dict_index_to_imdb.txt',
    './data/dict/movie_dict_id_to_imdb.txt',
    './data/dict/movie_id_to_sim_idx.txt'
)
tokenizer = Tokenizer('./data/dict/movie_dict_index_to_imdb.txt')
bert = BertRecommender(
    BertRecommenderConfig(
        16369,
        embedding_size=128,
        n_encoder_blocks=2,
        n_heads=4,
        feedforward_size=128 * 4,
        activation='relu',
        dropout_rate=0.2)
)

bert.load_weights('./data/model_weights/model')

sim_matrix = np.load('./data/sim_matrix/movie_sim.npy')
