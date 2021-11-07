from .cf_lib import getRecommendations_II


def pred(seq, sim_matrix, movie_id_idx_dict, topN=5):
    rec = getRecommendations_II(seq, sim_matrix, movie_id_idx_dict, topN)
    return list(rec.index)
