import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

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

    def SGD(self, lamda=0.01, alpha=0.03):
        '''
        lamda: regularization parameter
        alpha: learning rate
        >>> recsys = RecSys()
        >>> recsys.load_ratings("testcase_ratings.csv")
        #>>> recsys.load_ratings("../Dataset/ratings_Electronics_50.csv")
        #>>> recsys.SGD()
        '''

        # FIXME
        #u, s, vt = svds(self.util_mat)
        #rank = len(s)
        rank = 6

        n = self.util_mat.shape[0]
        m = self.util_mat.shape[1]
        U = np.random.rand(n, rank)
        V = np.random.rand(m, rank)

        #transform csr_matrix to coordinate matrix
        cx = self.util_mat.tocoo()
        for rnd in range(100):
            total_err = 0
            for count, (i, j, rating) in enumerate(zip(cx.row, cx.col, cx.data)):
                err = rating - np.dot(U[i], V[j].T)
                total_err += abs(err)
                U_i = U[i]
                V_j = V[j]
                U[i] = U_i + alpha * (err * V_j - lamda * U_i)
                V[j] = V_j + alpha * (err * U_i - lamda * V_j)
            print(total_err / (count + 1))

        #print(self.util_mat.todense())
        #print(np.dot(U, V.T))

    def biased_ALS(self, lamda=0.01):
        '''
        lamda: regularization parameter
        >>> recsys = RecSys()
        >>> recsys.load_ratings("testcase_ratings.csv")
        #>>> recsys.load_ratings("../Dataset/ratings_Electronics_50.csv")
        >>> recsys.biased_ALS()
        '''
        #FIXME u, s, vt = svds(self.util_mat)
        #rank = len(s)
        rank = 2

        n = self.util_mat.shape[0]
        m = self.util_mat.shape[1]
        U = np.random.rand(n, rank)
        V = np.random.rand(m, rank)
        betas = np.zeros((n, 1)) # users' biases
        gammas = np.zeros((m, 1)) # items' biases

        # TODO comment
        alpha = self.util_mat.count_nonzero() / n / m

        for rnd in range(20):
            aug_U = np.concatenate((np.atleast_2d(np.ones(n)).T, U), axis=1)
            for j in range(m):
                r_j = np.where(self.util_mat[:,j].todense() > 0, 1, 0)
                # FIXME need to understand the last dot term
                aug_V_j = np.dot(
                            np.linalg.inv(
                                np.dot(aug_U.T, aug_U) +
                                lamda * np.identity(rank + 1) +
                                alpha * np.dot((r_j * aug_U).T, aug_U)),
                            np.dot(
                                aug_U.T,
                                np.array(self.util_mat[:,j] - betas) * np.array(alpha * r_j + 1)))
                gammas[j] = aug_V_j[0]
                V[j,:] = aug_V_j[1:].T

            aug_V = np.concatenate((np.atleast_2d(np.ones(m)).T, V), axis=1)
            for i in range(n):
                r_i = np.where(self.util_mat[i].T.todense() > 0, 1, 0)
                # FIXME need to understand the last dot term
                aug_U_i = np.dot(
                            np.linalg.inv(
                                np.dot(aug_V.T, aug_V) +
                                lamda * np.identity(rank + 1) +
                                alpha * np.dot((r_i * aug_V).T, aug_V)),
                            np.dot(
                                aug_V.T,
                                np.array(self.util_mat[i].T - gammas) * np.array(alpha * r_i + 1)))
                betas[i] = aug_U_i[0]
                U[i,:] = aug_U_i[1:].T

#        for rnd in range(10):
#            # TODO comment
#            aug_U = np.concatenate((np.atleast_2d(np.ones(n)).T, U), axis=1)
#            aug_V = np.dot(
#                        np.linalg.inv(np.dot(aug_U.T, aug_U) + lamda * np.identity(rank + 1)),
#                        csr_matrix(aug_U.T).dot(self.util_mat - betas.repeat(m, axis=1))).T
#            gammas = aug_V[:,0]
#            V = aug_V[:,1:]
#
#            aug_V = np.concatenate((np.atleast_2d(np.ones(m)).T, V), axis=1)
#            aug_U = np.dot(
#                       np.linalg.inv(np.dot(aug_V.T, aug_V) + lamda * np.identity(rank + 1)),
#                       csr_matrix(aug_V.T).dot(self.util_mat.T - gammas.repeat(n, axis=1))).T
#            betas = aug_U[:,0]
#            U = aug_U[:,1:]
#
        print(self.util_mat.todense())
        print(np.dot(U, V.T) + betas.repeat(m, axis=1) + gammas.T.repeat(n, axis=0))
        print(betas)
        print(gammas)
