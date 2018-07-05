import numpy as np
from scipy.sparse import lil_matrix
from Base import Repo


class RecSysBaseLine(Repo):
    """
    A baseline recommander system which implements the item-to-item
    collaborative filtering.
    """

    def __init__(self):
        # the similarity matrix between items, this is used as a cache
        self.similarity_mat = None
        super().__init__()

    def load_ratings(self, arg, *args, **kwargs):
        super().load_ratings(arg, *args, **kwargs)
        num_items = self.util_mat.shape[1]
        self.similarity_mat = lil_matrix((num_items, num_items), dtype=float)

    def get_similarity(self, item_idx1, item_idx2):
        """
        Get the similarity between item1 and item2.
        To keep the sparsity of the similarity_mat, we always store the value
        in only the top half part of the matrix.
        """
        smaller_idx = item_idx1 if item_idx1 < item_idx2 else item_idx2
        larger_idx = item_idx2 if item_idx1 < item_idx2 else item_idx1

        return self.similarity_mat[smaller_idx, larger_idx]

    def set_similarity(self, item_idx1, item_idx2, sim):
        """
        Set the similarity between item1 and item2.
        To keep the sparsity of the similarity_mat, we always store the value
        in only the top half part of the matrix.
        """
        smaller_idx = item_idx1 if item_idx1 < item_idx2 else item_idx2
        larger_idx = item_idx2 if item_idx1 < item_idx2 else item_idx1

        self.similarity_mat[smaller_idx, larger_idx] = sim

    def predict_rating(self, target_user_idx, target_item_idx, N=50):
        """
        N: hyperparameter denoting the size of the most similar sets.
        Return the predicted rating of the user to the item.
        The prediction is done by item-to-item collaborative filtering using
        adjusted cosine similarity matrix.
        If the rating cannot be predicted due the item has not seen by any
        other users, the average rating of the user will be returned.

        >>> recsys = RecSysBaseLine()
        >>> recsys.load_ratings("testcase_ratings.csv")
        >>> user_idx = recsys.get_user_idx("U1")
        >>> item_idx = recsys.get_item_idx("I2")
        >>> "%.2f" %recsys.predict_rating(user_idx, item_idx)
        '4.00'
        >>> "%.2f" %recsys.predict_rating(user_idx, item_idx)
        '4.00'
        """

        # list all items which the active user has already rated
        rated_items = self.util_mat[target_user_idx].nonzero()[1]
        # list all users who has rated the target item
        rated_users = self.util_mat[:, target_item_idx].nonzero()[0]

        # effective items are the items that both target_user and at least one
        # other user has rated, where the other user has also rated the target
        # item. These are all the items we need to consider in the following
        # collaborative filtering.
        effective_items = []
        for user in rated_users:
            effective_items = np.union1d(
                    effective_items, self.util_mat[user].nonzero()[1])
        effective_items = np.intersect1d(effective_items, rated_items)

        most_similar_items = {}
        for item_idx in effective_items:
            item_idx = int(item_idx)

            # check if the similarity between the current item and the target
            # item has already been computed
            sim = self.get_similarity(item_idx, target_item_idx)
            if sim != 0:
                if sim > 0:
                    most_similar_items[item_idx] = sim
                continue

            # the set of all users who has rated both the current item and the
            # target item
            u_xy = np.intersect1d(
                self.util_mat[:, item_idx].nonzero()[0],
                self.util_mat[:, target_item_idx].nonzero()[0])

            # if the number of co-rating user is 1, the similarity score will
            # always be 1 regardless the ratings might be different. in this
            # case we don't consider the item as similar.
            if len(u_xy) <= 1:
                self.set_similarity(item_idx, target_item_idx, -1)
                continue

            # compute the similarity between the current item and the target
            # item, based on all the users in u_xy
            x = 0
            y = 0
            xy = 0
            for u in u_xy:
                r_ux = self.util_mat[u, item_idx]
                r_uy = self.util_mat[u, target_item_idx]
                r_u = self.get_avg_rating(u)

                xy += (r_ux - r_u) * (r_uy - r_u)
                x += np.square(r_ux - r_u)
                y += np.square(r_uy - r_u)
            sim = xy / np.sqrt(x) / np.sqrt(y)
            self.set_similarity(item_idx, target_item_idx, sim)

            # only consider the item with a positive similarity
            if sim > 0:
                most_similar_items[item_idx] = sim

        # compute the predicted rating of the target item from the ratings of
        # the N most similar items
        sorted_items = sorted(
                most_similar_items.items(), key=lambda x: x[1])[::-1]
        num = 0
        denom = 0
        for i, (item_idx, similarity) in enumerate(sorted_items):
            if i >= N:
                break
            num += similarity * self.util_mat[target_user_idx, item_idx]
            denom += abs(similarity)
        prediction = \
            num / denom if denom != 0 else self.get_avg_rating(target_user_idx)

        return prediction

    def predict_top_k_recomm(self, user_idx, k):
        """
        Return top k pairs of (item_idx, predicted_rating) according to the
        order of the predicted_rating.

        >>> recsys = RecSysBaseLine()
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
        for item_idx in rated_items:
            user_list = self.util_mat[:, item_idx].nonzero()[0]
            for user in user_list:
                candidate_items = np.union1d(
                        candidate_items, self.util_mat[user].nonzero()[1])
        candidate_items = np.setdiff1d(candidate_items, rated_items)

        predictions = []
        for item_idx in candidate_items:
            rating = self.predict_rating(user_idx, item_idx)
            if rating > 0:
                predictions.append((item_idx, rating))
        predictions.sort(key=lambda tup: tup[1], reverse=True)

        return predictions[0: k] if len(predictions) > k else predictions
