import numpy as np
from Base import Repo


class RecSysAdv(Repo):
    """
    An advanced recommender system which implements the matrix factorization
    collaborative filtering.
    """

    def __init__(self):
        self.U = None
        self.V = None
        super().__init__()

    def load_ratings(self, arg, *args, **kwargs):
        super().load_ratings(arg, *args, **kwargs)
        self.build_model()

    def build_model(self, rank=50):
        '''
        Use U (n x rank), V (m x rank) to estimate self.util_mat (n x m),
        such that the the difference between U * V_t and self.util_mat is
        minimized for the non-zero terms.
        U, V are initialized with random values, and then adjusted using
        stochastic gradient descent.
        '''

        lamda = 0.01  # regularization parameter
        alpha = 0.01  # learning rate
        min_shape = min(self.util_mat.shape)
        rank = rank if rank < min_shape else min_shape

        # n and m are computed from the user_ids and item_ids instead of the
        # shape of util_mat. This is because we may use exist mappings passed
        # from the evaluation system, which may contain additional users or
        # items that are not seen in the util_mat. And we should be able to
        # predict their ratings by matrix factorization.
        n = len(self.user_ids)
        m = len(self.item_ids)
        U = np.random.rand(n, rank)
        V = np.random.rand(m, rank)

        # transform csr_matrix to coordinate matrix
        cx = self.util_mat.tocoo()

        iterates = 0
        while True:
            # loop all ratings and adjust U and V accordingly
            avg_err = 0
            for count, data in enumerate(zip(cx.row, cx.col, cx.data)):
                i, j, rating = data
                err = rating - np.dot(U[i], V[j].T)
                avg_err += abs(err)
                U_i = U[i]
                V_j = V[j]
                U[i] = U_i + alpha * (err * V_j - lamda * U_i)
                V[j] = V_j + alpha * (err * U_i - lamda * V_j)

            # stop condition
            iterates += 1
            avg_err /= (count + 1)
            text = "Building model, err = {:.4f}".format(avg_err)
            print(text, end='\r', flush=True)
            if (iterates > 30) or (avg_err < 0.05):
                print()
                break

            # speed up a little bit in the later stage of iteration
            alpha = 0.02 if avg_err < 0.5 else 0.01

        self.U = U
        self.V = V

    def predict_rating(self, user_idx, item_idx):
        '''
        Return the predicted rating of the user to the item.
        The prediction is made by calculating U * V_t.
        '''
        rating = np.dot(self.U[user_idx], self.V[item_idx].T)
        return max(min(rating, 5), 0)

    def predict_top_k_recomm(self, user_idx, k):
        """
        Return top k pairs of (item_idx, predicted_rating) according to the
        order of the predicted_rating.

        >>> recsys = RecSysAdv()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> user_idx = recsys.get_user_idx("U1")
        >>> predictions = recsys.predict_top_k_recomm(user_idx, 2)
        >>> [(item, "%.2f" %sim) for (item, sim) in predictions]
        [(15.0, '4.00'), (16.0, '4.00')]
        """

        # list all items which the active user has already rated
        rated_items = self.util_mat[user_idx].nonzero()[1]

        # candidate items are items that can be recommanded to the target user.
        # It is generated from the following manner: First, find out all the
        # users that has rated the items the target user has also rated. Then,
        # union all the items these user has rated. Finally, exclude the items
        # the target user has already rated.
        candidate_items = []
        user_list = []
        for item_idx in rated_items:
            user_list = np.union1d(
                    user_list, self.util_mat[:, item_idx].nonzero()[0])
        for user in user_list:
            candidate_items = np.union1d(
                    candidate_items, self.util_mat[user].nonzero()[1])
        candidate_items = np.setdiff1d(candidate_items, rated_items)

        predictions = []
        for item_idx in candidate_items:
            rating = self.predict_rating(int(user_idx), int(item_idx))
            if rating > 0:
                predictions.append((item_idx, rating))
        predictions.sort(key=lambda tup: tup[1], reverse=True)

        return predictions[0: k] if len(predictions) > k else predictions
