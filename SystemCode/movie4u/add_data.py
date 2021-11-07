import pickle
import sqlite3

from imdbpie import Imdb


def execute_sql(s):
    con = sqlite3.connect('movie.db')
    with con:
        cur = con.cursor()
        cur.execute(s)


def single_quote(s):
    if len(s) == 0:
        return 'None'
    if s.find('\'') != -1:
        return s.replace("\'", "\'\'")
    else:
        return s


def get_text(title):
    try:
        text = title['plot']['outline']['text']
    except KeyError:
        try:
            text = title['plot']['text']
        except KeyError:
            text = 'No plot yet, we are so sorry about that.'

    return text


def get_video(movie_id):
    try:
        video = Imdb.get_title_videos(movie_id)['videos'][0]['encodings'][0]['play']
    except KeyError:
        video = ''

    return video


def get_image(title):
    try:
        image = title['base']['image']['url']
    except KeyError:
        image = ''

    return image


def get_rating(title):
    try:
        return title['ratings']['rating']
    except KeyError:
        return 0.0


def get_runtime(title):
    try:
        return title['base']['runningTimeInMinutes']
    except KeyError:
        return 0


def main():
    movie_list = []
    movie_genres = {}

    with open('data.csv') as f:
        for row in f.readlines()[1:]:
            columns = row.split(',')
            movie_id = columns[0].split('/')[4]
            genres = columns[1][:-1]
            movie_list.append(movie_id)
            movie_genres[movie_id] = genres

    imdb = Imdb()
    movie_count = 0
    failed = []

    for movie_id in movie_list:
        try:
            title = imdb.get_title(movie_id)
            sql = (
                '''INSERT INTO movie_movie VALUES (\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',\'{}\')'''.format(
                    movie_id,
                    single_quote(str(title['base']['title'])),
                    title['base']['year'],
                    get_runtime(title),
                    movie_genres[movie_id],
                    get_rating(title),
                    single_quote(get_image(title)),
                    single_quote(str(get_text(title))),
                    single_quote(str(get_video(movie_id)))
                ))
            execute_sql(sql)
            movie_count += 1
            print("Insert movie: " + movie_id, movie_count)
        except KeyboardInterrupt:
            with open(f'failed.data', 'wb') as f:
                pickle.dump(failed, f)
        except Exception as e:
            print('Movie Insert Failure: ' + movie_id, e)
            failed.append({'movie_id': movie_id, 'index': iii - 1, 'error': e.__str__()})
            continue
        print('\n')

    with open('failed.data', 'wb') as f:
        pickle.dump(failed, f)


if __name__ == '__main__':
    main()
