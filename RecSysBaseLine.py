import numpy as np
from scipy.sparse import csr_matrix
from RecSys import RecSys

class RecSysBaseLine(RecSys):
    def __init__(self):
        super().__init__()

    def predict_rating(self, user_idx, item_idx, adjusted=False):
        """
        >>> recsys = RecSysBaseLine()
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
        return pairs of (item_idx, predicted_rating)

        >>> recsys = RecSysBaseLine()
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
