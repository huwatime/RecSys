import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from RecSys import RecSys

class AdvRecSys(RecSys):
    def __init__(self):
        super().__init__()

    def SGD(self, lamda=0.01, alpha=0.03):
        '''
        lamda: regularization parameter
        alpha: learning rate
        >>> recsys = AdvRecSys()
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
        >>> recsys = AdvRecSys()
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
