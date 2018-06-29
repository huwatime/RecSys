import numpy as np
from scipy.sparse import csr_matrix

class RecSys():
    def __init__(self):
        self.util_mat = None  # the user-item utility matrix
        self.user_rating_means = []  # users' rating means
        self.user_id_map = {}  # the user-id to user-index map
        self.item_id_map = {}  # the item-id to item-index map
        self.user_ids = [] # map user-idx to user-id
        self.item_ids = [] # map item-idx to item-id

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

    def get_user_id(self, user_idx):
        """
        map user's idx to id
        """
        return self.user_ids[int(user_idx)]

    def get_item_id(self, item_idx):
        """
        map item's idx to id
        """
        return self.item_ids[int(item_idx)]

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
                    self.user_ids.append(user_id)
                # if the item_id is seen for the first time, assign an index
                # (index starts from 0) and store into item_id_map
                if (item_id not in self.item_id_map):
                    self.item_id_map[item_id] = item_idx
                    item_idx = item_idx + 1
                    self.item_ids.append(item_id)

                users.append(self.user_id_map[user_id])
                items.append(self.item_id_map[item_id])
                ratings.append(float(rating))

        # generate user-item utility matrix (using sparse matrix)
        self.util_mat = csr_matrix((ratings, (users, items)), dtype=float)

        # compute rating mean for each user
        self.user_rating_means = np.bincount(users, weights=ratings)\
            / np.bincount(users)

