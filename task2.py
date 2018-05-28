import numpy as np
from scipy.sparse import csr_matrix

CSV_PATH = '../Dataset/ratings_Electronics_50.csv'


class RecSys:
    def __init__(self):
        self.util_mat = None  # the user-item utility matrix
        self.user_rating_means = []  # users' rating means
        self.user_id_map = {}  # the user-id to user-index map
        self.item_id_map = {}  # the item-id to item-index map

    def get_user_idx(self, user_id):
        """
        map user's id to index
        """
        return self.user_id_map[user_id]

    def get_item_idx(self, item_id):
        """
        map item's id to index
        """
        return self.item_id_map[item_id]

    def load_ratings(self, ratings_file):
        """
        Load ratings from the given file.
        The expected format of the file is one rating per line,
        in the format <User_ID>,<Item_ID>,<Rating>,<Time>
        Each line from the given file will be split (at ",") into 4 tokens:
        user_id, item_id, rating and timestamp.
        First seen user_id or item_id will be assign an respective index and
        store into map for further usage.
        All ratings from the given file will be stored in a sparse matrix
        (util_mat), where row indices will be the same as user indices
        and column indices will be the same to item indices.
        Finally, the average rating for every user will be computed and stored
        in a list (user_rating_means).

        >>> recsys = RecSys()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> print(recsys.user_id_map)
        {'U1': 0, 'U2': 1, 'U3': 2, 'U4': 3, 'U5': 4}
        >>> print(recsys.item_id_map)
        {'I1': 0, 'I2': 1, 'I3': 2, 'I4': 3, 'I5': 4, 'I6': 5}
        >>> print(recsys.util_mat.todense())
        [[5. 2. 4. 0. 0. 0.]
         [1. 0. 3. 2. 0. 0.]
         [0. 0. 4. 0. 0. 0.]
         [0. 4. 5. 3. 4. 4.]
         [0. 2. 0. 4. 3. 0.]]
        >>> [("%.2f" %mean) for mean in recsys.user_rating_means]
        ['3.67', '2.00', '4.00', '4.00', '3.00']
        """
        user_idx = 0  # user index starts from 0
        item_idx = 0  # item index starts from 0
        users = []
        items = []
        ratings = []

        with open(ratings_file, 'r') as f:
            for line in f:
                # split each line (at ",") into tokens
                user_id, item_id, rating, timestamp = line.split(",")

                # if the user_id is seen for the first time, assign an index
                # (index starts from 0) and store into user_id_map
                if (user_id not in self.user_id_map):
                    self.user_id_map[user_id] = user_idx
                    user_idx = user_idx + 1
                # if the item_id is seen for the first time, assign an index
                # (index starts from 0) and store into item_id_map
                if (item_id not in self.item_id_map):
                    self.item_id_map[item_id] = item_idx
                    item_idx = item_idx + 1

                users.append(self.user_id_map[user_id])
                items.append(self.item_id_map[item_id])
                ratings.append(float(rating))

        # generate user-item utility matrix (using sparse matrix)
        self.util_mat = csr_matrix((ratings, (users, items)), dtype=float)

        # compute rating mean for each user
        self.user_rating_means = np.bincount(users, weights=ratings)\
            / np.bincount(users)

    def predict_rating(self, user_idx, item_idx, adjusted=False):
        """
        >>> recsys = RecSys()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> user_idx = recsys.get_user_idx("U1")
        >>> item_idx = recsys.get_item_idx("I4")
        >>> ["%.2f" %recsys.predict_rating(user_idx, item_idx)]
        ['3.06']
        """
        # list all items which the active user has already rated
        item_list = self.util_mat[user_idx].nonzero()[1]
        # list all users who has rated the target item
        user_list = self.util_mat[:, item_idx].nonzero()[0]

        # candidate items
        cand_item_list = []

        for user in user_list:
            cand_item_list = np.union1d(cand_item_list,
                                        self.util_mat[user].nonzero()[1])
        cand_item_list = np.intersect1d(cand_item_list, item_list)

        sim = []
        l = []
        for item in cand_item_list:
            # the set of all users co-rating both item
            u_xy = np.intersect1d(
                self.util_mat[:, item].nonzero()[0],
                self.util_mat[:, item_idx].nonzero()[0])
            # if the number of co-rating user is 1,
            # the similarity score will always be 1
            # regardless the ratings might be different
            if len(u_xy) <= 1:
                continue
            x = 0
            y = 0
            xy = 0
            for u in u_xy:
                r_ux = self.util_mat[u, item]
                r_uy = self.util_mat[u, item_idx]
                r_u = self.user_rating_means[u] if adjusted else 0

                xy += (r_ux - r_u) * (r_uy - r_u)
                x += np.square(r_ux - r_u)
                y += np.square(r_uy - r_u)
            sim.append(xy / np.sqrt(x) / np.sqrt(y))
            l.append(item)

        if len(l) == 0:
            return 0

        num = 0
        deno = 0
        for i in range(0, len(l)):
            num += sim[i] * self.util_mat[user_idx, l[i]]
            deno += abs(sim[i])
        prediction = num / deno

        return prediction

    def predict_top_k_recomm(self, user_idx, k):
        """
        >>> recsys = RecSys()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> user_idx = recsys.get_user_idx("U1")
        >>> predictions = recsys.predict_top_k_recomm(user_idx, 2)
        >>> [(item, "%.2f" %sim) for (item, sim) in predictions]
        [(3.0, '3.06'), (4.0, '2.00')]
        """

        # list all items which the active user has already rated
        item_list = self.util_mat[user_idx].nonzero()[1]
        cand_item_list = []

        for item in item_list:
            user_list = self.util_mat[:, item].nonzero()[0]
            for user in user_list:
                cand_item_list = np.union1d(cand_item_list,
                                            self.util_mat[user].nonzero()[1])
        cand_item_list = np.setdiff1d(cand_item_list, item_list)

        predictions = []
        for item in cand_item_list:
            predictions.append((item, self.predict_rating(user_idx, item)))
        predictions.sort(key=lambda tup: tup[1], reverse=True)

        return predictions[0: k]
