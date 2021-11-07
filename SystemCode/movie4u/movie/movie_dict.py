class MovieDict(object):
    def __init__(self, index_to_imdb_path, id_to_imdb_path, id_to_sim_idx_path):
        self.imdb_to_index = {}
        self.index_to_imdb = {}

        self.imdb_to_id = {}
        self.id_to_imdb = {}

        self.id_to_sim_idx = {}
        self.sim_idx_to_id = {}

        self.build_dict(index_to_imdb_path, id_to_imdb_path, id_to_sim_idx_path)

    def build_dict(self, index_to_imdb_path, id_to_imdb_path, id_to_sim_idx_path):
        with open(index_to_imdb_path, 'r') as f:
            for line in f:
                index, imdb = line.strip().split(' ')
                self.imdb_to_index[imdb] = int(index)
                self.index_to_imdb[int(index)] = imdb

        with open(id_to_imdb_path, 'r') as f:
            for line in f:
                iid, imdb = line.strip().split(' ')
                self.imdb_to_id[imdb] = int(iid)
                self.id_to_imdb[int(iid)] = imdb

        with open(id_to_sim_idx_path, 'r') as f:
            for line in f:
                iid, idx = line.strip().split(' ')
                self.id_to_sim_idx[int(iid)] = int(idx)
                self.sim_idx_to_id[int(idx)] = int(iid)

    def imdb_to_sim_idx(self, seq):
        result = []
        for imdb in seq:
            try:
                iid = self.imdb_to_id[imdb]
                idx = self.id_to_sim_idx[iid]
            except KeyError:
                continue
            result.append(idx)

        return result

    def ids_to_imdb(self, seq):
        seq = [self.id_to_imdb[iid] for iid in seq]

        return seq
