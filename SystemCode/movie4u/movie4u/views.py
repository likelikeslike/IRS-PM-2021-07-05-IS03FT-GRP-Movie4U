from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_protect
from movie.models import *
import operator
import random
import numpy as np
from movie.initializer import search_index, movie_dicts, sim_matrix, bert, tokenizer
from model.utils import predict
from model.cf_recommender import pred

@csrf_protect
def index(request):
    data = {}
    movie_dict = search_index.data_in_memory['movie_dict']
    if request.user.is_authenticated:
        data = {'username': request.user.get_username()}
    popular_movies = Popularity.objects.all().order_by('-weight')
    popular = []
    for movie in popular_movies[:9]:
        try:
            popular.append({'movieid': movie.movieid_id, 'poster': movie_dict[movie.movieid_id].poster})
        except:
            continue
    data['popular'] = popular
    popular_movie_list = [movie_dict[movie.movieid_id] for movie in popular_movies[:5]]
    data['recommendation'] = get_recommendation(request, popular_movie_list)
    return render(request, 'base.html', data)


def get_recommendation(request, popular_movie_list):
    result = []
    movie_dict = search_index.data_in_memory['movie_dict']
    added_movie_list = []
    rec = []
    if request.user.is_authenticated:
        username = request.user.get_username()
        watched_movies = set([movie_dict[movie.movieid_id] for movie in Seen.objects.filter(username=username)] +
                             [movie_dict[movie.movieid_id] for movie in Expect.objects.filter(username=username)])

        if len(watched_movies) > 0:
            # item based recommendation
            watched_movies_imdb = [movie.movieid for movie in watched_movies]
            watched_movies_idx = movie_dicts.imdb_to_sim_idx(watched_movies_imdb)
            ipt = np.array([np.nan for _ in range(len(movie_dicts.id_to_sim_idx))])
            ipt[watched_movies_idx] = 4
            cf_rec = pred(ipt, sim_matrix, movie_dicts.id_to_sim_idx, 3)
            cf_rec = movie_dicts.ids_to_imdb(cf_rec)
            cf_rec_movie = [Movie.objects.filter(movieid=iid)[0] for iid in cf_rec]
            # bert recommendation
            bert_rec = predict(bert, watched_movies_imdb, tokenizer, 3)
            bert_rec_movie = [Movie.objects.filter(movieid=iid)[0] for iid in bert_rec]

            rec = set(cf_rec_movie + bert_rec_movie) - set(popular_movie_list) - set(watched_movies)

        unwatched_movies = set(search_index.data_in_memory['movie_list']) - watched_movies - set(popular_movie_list)
        genre_stats = {}
        for movie in watched_movies:
            for genre in movie.genres.split('|'):
                genre_stats[genre] = genre_stats.get(genre, 0) + 1
        movie_score = {}
        for movie in unwatched_movies:
            movie_score[movie.movieid] = movie.rate
            for genre in movie.genres.split('|'):
                try:
                    movie_score[movie.movieid] += genre_stats.get(genre, 0) / len(watched_movies)
                except ZeroDivisionError:
                    continue
        sorted_list = sorted(movie_score.items(), key=operator.itemgetter(1), reverse=True)
        for item in sorted_list:
            movie = movie_dict[item[0]]
            result.append({'movieid': movie.movieid, 'poster': movie.poster})
            added_movie_list.append(movie)
            if len(result) == 8:
                break

    sorted_list = sorted(search_index.data_in_memory['movie_rating'].items(), key=operator.itemgetter(1), reverse=True)
    for item in sorted_list:
        movie = movie_dict[item[0]]
        if movie not in popular_movie_list and movie not in added_movie_list:
            result.append({'movieid': movie.movieid, 'poster': movie.poster})
        if len(result) == 10:
            break

    if len(result) > 3:
        return [result[i] for i in random.sample(range(len(result)), 3)] + list(rec)
    else:
        return result + list(rec)

