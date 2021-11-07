import re
from .models import Movie


class Index(object):
    data_in_memory = {'movie_dict': {}, 'movie_list': [],
                      'movie_rating': {}, 'movie_genre': {}}
    movie_index = {}

    def __init__(self):
        self.__load_data_from_db()
        self.__build_index()

    def __load_data_from_db(self):
        for movie in Movie.objects.all():
            self.data_in_memory['movie_dict'][movie.movieid] = movie
            self.data_in_memory['movie_list'].append(movie)
            self.data_in_memory['movie_rating'][movie.movieid] = movie.rate
            genres = movie.genres.split('|')
            for genre in genres:
                if genre not in self.data_in_memory['movie_genre']:
                    self.data_in_memory['movie_genre'][genre] = set()
                self.data_in_memory['movie_genre'][genre].add(movie.movieid)

    def __build_index(self):
        for movie in self.data_in_memory['movie_list']:
            for term in self.tokenize(movie.title):
                for permuted_term in self.permute(term):
                    if permuted_term not in self.movie_index:
                        self.movie_index[permuted_term] = set()
                    self.movie_index[permuted_term].add(movie.movieid)

    def search_movie(self, query_string):
        high_matching_movies, middle_matching_movies, low_matching_movies = set(), set(), set()
        for token in self.tokenize(query_string):
            start_with_token = self.rotate(token + "*")
            end_with_token = self.rotate("*" + token)
            movie_result = set()
            for movie in self.search_index(self.movie_index, [start_with_token, end_with_token]):
                movie_result.add(movie)
            wild_tokens = self.add_wild_card(token)
            for movie in self.search_index(self.movie_index, [self.rotate(t) for t in wild_tokens]):
                low_matching_movies.add(movie)
            if len(high_matching_movies) == 0:
                high_matching_movies = high_matching_movies.union(movie_result)
            else:
                high_matching_movies = high_matching_movies.intersection(movie_result)
            middle_matching_movies = middle_matching_movies.union(movie_result)
        middle_matching_movies = middle_matching_movies - high_matching_movies
        low_matching_movies = low_matching_movies - high_matching_movies - middle_matching_movies
        return (sorted(high_matching_movies, key=self.get_movie_rating, reverse=True) +
                sorted(middle_matching_movies, key=self.get_movie_rating, reverse=True) +
                sorted(low_matching_movies, key=self.get_movie_rating, reverse=True))

    def search_suggest(self, query_string):
        movie_flag = False
        high_matching_movies, middle_matching_movies, low_matching_movies = set(), set(), set()
        for token in self.tokenize(query_string):
            start_with_token = self.rotate(token + "*")
            end_with_token = self.rotate("*" + token)
            movie_result = set()
            for movie in self.search_index(self.movie_index, [start_with_token, end_with_token]):
                movie_result.add(movie)
            if len(high_matching_movies) > 2:
                movie_flag = True
            if movie_flag:
                continue
            wild_tokens = self.add_wild_card(token)
            if not movie_flag:
                for movie in self.search_index(self.movie_index, [self.rotate(t) for t in wild_tokens]):
                    low_matching_movies.add(movie)

            if len(high_matching_movies) == 0:
                high_matching_movies = high_matching_movies.union(movie_result)
            else:
                high_matching_movies = high_matching_movies.intersection(movie_result)

        middle_matching_movies = middle_matching_movies - high_matching_movies
        low_matching_movies = low_matching_movies - high_matching_movies - middle_matching_movies

        movie_result = sorted(high_matching_movies, key=self.get_movie_rating, reverse=True)
        if len(movie_result) < 3:
            movie_result += sorted(middle_matching_movies, key=self.get_movie_rating, reverse=True)
        if len(movie_result) < 3:
            movie_result += sorted(low_matching_movies, key=self.get_movie_rating, reverse=True)

        return movie_result

    def get_movie_rating(self, movie_id):
        return self.data_in_memory['movie_dict'][movie_id].rate

    @staticmethod
    def search_index(index_dict, token_list):
        result = set()
        for token in token_list:
            for key in index_dict:
                if key.startswith(token[:-1]):
                    result = result.union(index_dict[key])
        return list(result)

    @staticmethod
    def tokenize(text):
        clean_string = re.sub('[^a-z0-9 ]', ' ', text.lower())
        tokens = clean_string.split()
        return tokens

    @staticmethod
    def permute(term):
        x = term + "$"
        return [x[i:] + x[:i] for i in range(len(x))]

    @staticmethod
    def rotate(term):
        x = term + "$"
        if "*" not in term:
            return x
        n = x.index("*") + 1
        return x[n:] + x[:n]

    @staticmethod
    def add_wild_card(term):
        tokens = []
        n = len(term)
        for i in range(1, n):
            tokens.append(term[:i] + "*" + term[i:])
        return tokens
