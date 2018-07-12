import math
import random
import time
import numpy as np
from Base import Repo
from Base import EvaMatrix


class EvaSys(Repo):
    """
    An evaluation system. Given rating file(s), it evaluates the performance of
    a recommander system using the k-fold manner. Namely, split all rating data
    into k folds. In each round, take one fold as test data while the other
    k-1 folds as training data. Calculate the performance of the trained model
    based on measurement matrics including RMSE, MAE, P@M, R@M, MRR@M, nDCG@M,
    and time@M. Finally, take the average of the k models as the performance of
    the recommander system.
    """

    def __init__(self):
        self.split_rating_files = []  # Array of the split filenames
        self.total_rating_file = ""  # Filename that contains all the ratings
        self.num_fold = 0
        super().__init__()

    def load_split_ratings(self, split_files):
        """
        Load the pre-split files.
        split_files: filenames in array form
        """

        assert len(split_files) > 1, \
            "Bad parameter, split_files must be an array of size >= 2"

        self.split_rating_files = split_files
        self.num_fold = len(self.split_rating_files)
        self.total_rating_file = "total_rating.csv"

        # Combine all splits into one rating file and then load ratings
        with open(self.total_rating_file, 'w') as outfile:
            for f in self.split_rating_files:
                    with open(f) as infile:
                        for line in infile:
                            outfile.write(line)
        self.load_ratings(self.total_rating_file)

    def load_total_ratings(self, num_fold, filename):
        """
        Load the ratings file and split it into num_fold folds.
        >>> evasys = EvaSys()
        >>> evasys.load_total_ratings(3, "testcase_ratings.csv")
        """

        self.total_rating_file = filename
        self.num_fold = num_fold
        self.load_ratings(self.total_rating_file)

        # generate filenames for split files
        for fold in range(num_fold):
            self.split_rating_files.append(
                    "ratings_split_" + str(fold + 1) + ".csv")

        # generate an array for assignment
        # each line in the rating_file will be assigned to different files
        # according to the split map
        q, r = divmod(self.util_mat.nnz, num_fold)
        split_map = np.concatenate(
                (np.tile(np.arange(num_fold), q), np.arange(r)))

        num_users = self.util_mat.shape[0]
        begin = 0
        end = 0
        for user_idx in range(num_users):
            # locate the region for shuffling
            end = begin + self.util_mat[user_idx].count_nonzero()
            # shuffle each user's ratings
            np.random.shuffle(split_map[begin:end])
            begin = end

        # write each line in the rating_file to different split file
        # according to the assignment array
        fw = []
        for fold in range(num_fold):
            fw.append(open(self.split_rating_files[fold], 'w'))
        with open(self.total_rating_file, 'r') as fr:
            for i, line in enumerate(fr):
                fw[split_map[i]].write(line)

    def evaluate(self, rec_sys, positions, use_all_user=True, num_user=10):
        """
        Evaluate the given recommander system rec_sys in the k-fold manner.
        Return the measurement matrics including RMSE, MAE, P@M, R@M, MRR@M,
        nDCG@M, and time@M.
        positions is an array specifying the Ms.
        use_all_user is a boolean to control if we evaluate on all users. If
        false, choose num_user users randomly to evaluate.
        """

        user_list = []
        if not use_all_user:
            print("\n=============== Evaluate on the following {} users:"
                  .format(num_user))
            for i in range(num_user):
                user_idx = random.randint(0, len(self.user_ids)-1)
                user_list.append(user_idx)
                user_id = self.get_user_id(user_idx)
                print("user {}: {}".format(i+1, user_id))

        result_all_model = EvaMatrix(positions)
        for fold in range(self.num_fold):
            print("\n=============== Fold", fold+1)
            training_file, test_file = self.generate_train_test_files(fold)
            result = self.evaluate_core(
                    training_file, test_file, rec_sys, positions, user_list)
            result_all_model.accumulate(result)
        result_all_model.avg(self.num_fold, self.num_fold)
        print("\n=============== Matrics Avg and Total Time")
        result_all_model.print_data()

        return result_all_model

    def generate_train_test_files(self, k):
        """
        k: zero-based index
        Return a training_file and a test_file, where the k-th file in
        self.split_rating_files is the test file, and the other files are
        concatenating together as the training file.
        """

        training_file_name = 'training_file_' + str(k+1) + '.csv'

        with open(training_file_name, 'w') as training_file:
            for idx, filename in enumerate(self.split_rating_files):
                if (idx == k):
                    test_file_name = filename
                else:
                    with open(filename) as infile:
                        for line in infile:
                            training_file.write(line)

        return training_file_name, test_file_name

    def evaluate_core(
            self, training_file, test_file, rec_sys, positions, user_list):
        start_time = time.time()
        rec_sys.load_ratings(
                training_file,
                True,
                self.user_id_map,
                self.item_id_map,
                self.user_ids,
                self.item_ids)

        test_data = Repo()
        test_data.load_ratings(
                test_file,
                True,
                self.user_id_map,
                self.item_id_map,
                self.user_ids,
                self.item_ids)

        if len(user_list) == 0:
            user_list = range(test_data.util_mat.shape[0])
        num_users = len(user_list)
        num_ratings = 0
        result_all_user = EvaMatrix(positions)

        # Loop all users in test_data
        for user_cnt, user_idx in enumerate(user_list):
            text = "Computing User {}/{} ...".format(user_cnt + 1, num_users)
            print(text, flush=True, end='\r')
            result = EvaMatrix(positions)
            positive_items = {}

            # Get the user's average rating (based on all the data)
            score_avg = self.get_avg_rating(user_idx)

            # Loop all items the user has rated
            rated_items = test_data.util_mat[user_idx].nonzero()[1]
            for item_idx in rated_items:
                # Binarize the score
                score_true = self.util_mat[user_idx, item_idx]
                if score_true >= score_avg:
                    positive_items[item_idx] = 1

                # Accuamulate RMSE and MAE
                score_pred = rec_sys.predict_rating(user_idx, item_idx)
                score_diff = abs(score_pred - score_true)
                result.rmse += np.power(score_diff, 2)
                result.mae += score_diff
                num_ratings += 1

            if len(positive_items) == 0:
                result_all_user.accumulate(result)
                continue

            # Loop top k recommanded items to the user
            largest_k = np.amax(positions)
            predictions = rec_sys.predict_top_k_recomm(user_idx, largest_k)
            relevant_count = np.zeros(len(positions))
            dcg = np.zeros(len(positions))
            idcg = np.zeros(len(positions))
            for item_count, (item_idx, score_pred) in enumerate(predictions):
                # Accumulate IDCG according to k
                for i in range(len(positions)):
                    if item_count < positions[i]:
                        idcg[i] += 1. / math.log(1 + item_count + 1)

                # Handle a hit
                if item_idx in positive_items:
                    for i in range(len(positions)):
                        if item_count < positions[i]:
                            relevant_count[i] += 1
                            dcg[i] += 1. / math.log(1 + item_count + 1)

                            # Update MRR@K directly
                            if result.mrr_at_k[i] == 0:
                                result.mrr_at_k[i] = 1./(item_count + 1)

            # Comppute XX@K
            result.p_at_k = relevant_count / positions
            result.r_at_k = relevant_count / len(positive_items)
            result.ndcg_at_k = dcg / idcg if (idcg != 0).all() else 0

            result_all_user.accumulate(result)

        result_all_user.avg(num_ratings, num_users)
        result_all_user.rmse = np.sqrt(result_all_user.rmse)
        result_all_user.time = time.time() - start_time
        print()
        result_all_user.print_data()
        return result_all_user
