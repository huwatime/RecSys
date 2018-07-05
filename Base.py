import numpy as np
from scipy.sparse import csr_matrix


class Repo():
    """
    A base class for a recommander/evaluation system.
    After calling load_ratings(), the user-item rating matrix is generated.
    Functions for mapping between ID and index are also given.
    """

    def __init__(self):
        self.util_mat = None  # the user-item utility(rating) matrix
        self.user_rating_means = []  # users' rating means
        self.user_id_map = {}  # map user_id to user_idx
        self.item_id_map = {}  # map item_id to item_idx
        self.user_ids = []  # map user_idx to user_id
        self.item_ids = []  # map item_idx to item_id

    def get_user_idx(self, user_id):
        """
        >>> recsys = Repo()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> recsys.get_user_idx('U8')
        7
        """
        return self.user_id_map[user_id]

    def get_item_idx(self, item_id):
        """
        >>> recsys = Repo()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> recsys.get_item_idx('I3')
        18
        """
        return self.item_id_map[item_id]

    def get_user_id(self, user_idx):
        """
        >>> recsys = Repo()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> recsys.get_user_id(7)
        'U8'
        """
        return self.user_ids[int(user_idx)]

    def get_item_id(self, item_idx):
        """
        >>> recsys = Repo()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> recsys.get_item_id(15)
        'I18'
        """
        return self.item_ids[int(item_idx)]

    def get_avg_rating(self, user_idx=-1, user_id=""):
        """
        Return the average rating of the given user.
        You can specify either its ID or its index.

        If both are specified, ID will be used.
        >>> recsys = Repo()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> "{:.2f}".format(recsys.get_avg_rating(user_id="U2"))
        '2.70'
        >>> "{:.2f}".format(recsys.get_avg_rating(user_idx=1))
        '2.70'
        >>> "{:.2f}".format(recsys.get_avg_rating(3, "U2"))
        '2.70'
        """
        assert (user_id in self.user_id_map) or \
            (user_idx >= 0 and user_idx < len(self.user_ids)), \
            "Bad parameter. user_id = [{}], user_idx = [{}]".format(
                    user_id, user_idx)

        if (user_id in self.user_id_map):
            idx = self.get_user_idx(user_id)
            return self.user_rating_means[idx]
        else:
            return self.user_rating_means[user_idx]

    def load_ratings(
            self,
            ratings_file,
            use_exist_mapping=False,
            user_id_map={},
            item_id_map={},
            user_ids=[],
            item_ids=[]
            ):
        """
        Load ratings from the given file.
        The expected format of the file is one rating per line,
        in the format <User_ID>,<Item_ID>,<Rating>,<Time>[,<Fold>]
        Each line from the given file will be split (at ",") into tokens.

        If the mappings between ID and index are not given, user_ids and
        item_ids will be assigned to respective indexes in the order they are
        seen. The mappings are stored for further usage.

        All ratings from the given file will be stored in a sparse matrix
        (util_mat), where row indices will be the same as user indices and
        column indices will be the same to item indices.

        Finally, the average rating for every user will be computed and stored
        in a list (user_rating_means).

        >>> recsys = Repo()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> print(recsys.util_mat.todense())
        [[2. 2. 1. 4. 4. 2. 2. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [3. 0. 3. 5. 0. 0. 2. 0. 0. 0. 2. 2. 1. 2. 2. 5. 0. 0. 0. 0.]
         [0. 0. 3. 0. 4. 0. 2. 2. 2. 0. 0. 2. 1. 2. 2. 0. 2. 2. 0. 0.]
         [0. 0. 0. 5. 0. 0. 2. 1. 0. 0. 1. 0. 1. 2. 0. 0. 4. 3. 2. 0.]
         [2. 0. 1. 4. 5. 0. 0. 1. 3. 0. 2. 2. 1. 0. 3. 0. 0. 0. 0. 0.]
         [3. 5. 0. 0. 5. 3. 0. 2. 0. 0. 0. 0. 1. 2. 0. 0. 0. 2. 0. 2.]
         [4. 0. 4. 0. 0. 4. 3. 0. 0. 4. 4. 3. 1. 2. 4. 0. 3. 0. 3. 0.]
         [2. 0. 0. 4. 4. 0. 0. 1. 0. 0. 0. 2. 0. 0. 0. 5. 3. 0. 2. 0.]]
        >>> [("%.2f" %mean) for mean in recsys.user_rating_means]
        ['1.91', '2.70', '2.18', '2.33', '2.40', '2.78', '3.25', '2.88']
        """
        user_idx = 0  # user index starts from 0
        item_idx = 0  # item index starts from 0
        users = []
        items = []
        ratings = []

        if use_exist_mapping:
            self.user_id_map = user_id_map
            self.item_id_map = item_id_map
            self.user_ids = user_ids
            self.item_ids = item_ids

        with open(ratings_file, 'r') as f:
            for line in f:
                # split each line (at ",") into tokens
                splits = line.split(",")
                user_id, item_id, rating = splits[0:3]

                if not use_exist_mapping:
                    # if the user_id is seen for the first time, assign an
                    # index (index starts from 0) and store into user_id_map
                    if (user_id not in self.user_id_map):
                        self.user_id_map[user_id] = user_idx
                        user_idx = user_idx + 1
                        self.user_ids.append(user_id)
                    # if the item_id is seen for the first time, assign an
                    # index (index starts from 0) and store into item_id_map
                    if (item_id not in self.item_id_map):
                        self.item_id_map[item_id] = item_idx
                        item_idx = item_idx + 1
                        self.item_ids.append(item_id)

                users.append(self.user_id_map[user_id])
                items.append(self.item_id_map[item_id])
                ratings.append(float(rating))

        # generate user-item utility matrix (using sparse matrix)
        self.util_mat = csr_matrix(
                (ratings, (users, items)),
                dtype=float,
                shape=(len(self.user_ids), len(self.item_ids)))

        # compute the avarage rating of each user
        self.user_rating_means = np.bincount(users, weights=ratings)\
            / np.bincount(users)


class EvaMatrix():
    """
    A Class to store the values of all the evaluation matrics.
    Class-wise addition and division functions are provided.
    """

    def __init__(self, positions):
        self.rmse = 0
        self.mae = 0
        self.positions = positions  # An array specifing the Ks
        self.p_at_k = np.zeros(len(positions))
        self.r_at_k = np.zeros(len(positions))
        self.mrr_at_k = np.zeros(len(positions))
        self.ndcg_at_k = np.zeros(len(positions))
        self.time = 0

    def accumulate(self, mat):
        """
        Add all members respectively in mat to self.

        >>> positions = [1, 2, 5, 10]
        >>> eva1 = EvaMatrix(positions)
        >>> eva1.rmse = 10.3
        >>> eva1.mrr_at_k = [0.4, 0.6, 0.2, 13]
        >>> eva2 = EvaMatrix(positions)
        >>> eva2.rmse = 2.2
        >>> eva2.p_at_k = [1, 2, 3, 4]
        >>> eva2.mrr_at_k = [1.2, 1.3, 12, 0]
        >>> eva1.accumulate(eva2)
        >>> eva1.rmse
        12.5
        >>> [("%.1f" %val) for val in eva1.p_at_k]
        ['1.0', '2.0', '3.0', '4.0']
        >>> [("%.1f" %val) for val in eva1.mrr_at_k]
        ['1.6', '1.9', '12.2', '13.0']
        """

        assert (mat.positions == self.positions), "Positions not align"
        self.rmse += mat.rmse
        self.mae += mat.mae
        self.p_at_k = np.add(self.p_at_k, mat.p_at_k)
        self.r_at_k = np.add(self.r_at_k, mat.r_at_k)
        self.mrr_at_k = np.add(self.mrr_at_k, mat.mrr_at_k)
        self.ndcg_at_k = np.add(self.ndcg_at_k, mat.ndcg_at_k)
        self.time += mat.time

    def avg(self, denom1, denom2):
        """
        Take average on all members, where rmse and mat are divided by denom1,
        the other XX@K metrics are divided by denom2, and time is NOT divided.

        >>> positions = [1, 2, 5, 10]
        >>> eva1 = EvaMatrix(positions)
        >>> eva1.rmse = 10.3
        >>> eva1.mrr_at_k = np.array([0.4, 0.6, 0.2, 13])
        >>> eva1.ndcg_at_k = np.array([25, 18, 7, 0.9])
        >>> eva1.avg(0.1, 10)
        >>> eva1.rmse
        103.0
        >>> [("%.2f" %val) for val in eva1.mrr_at_k]
        ['0.04', '0.06', '0.02', '1.30']
        >>> [("%.2f" %val) for val in eva1.ndcg_at_k]
        ['2.50', '1.80', '0.70', '0.09']
        """
        self.rmse /= denom1
        self.mae /= denom1
        self.p_at_k /= denom2
        self.r_at_k /= denom2
        self.mrr_at_k /= denom2
        self.ndcg_at_k /= denom2
        self.time /= 1.

    def print_data(self):
        np.set_printoptions(precision=2)
        print("RMSE = {:.2f}\nMAE  = {:.2f}".format(self.rmse, self.mae))
        print(
            "P@K    = {}\nR@K    = {}\nMRR@K  = {}\nNDCG@K = {}".format(
                self.p_at_k, self.r_at_k, self.mrr_at_k, self.ndcg_at_k))
        print("time = {:.2f} Sec".format(self.time))
