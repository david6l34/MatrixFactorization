import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.decomposition
import scipy.sparse as ss
import matplotlib.pyplot as plt

df = pd.read_csv("data/ml-latest-small/ratings.csv")
df['timestamp'] = pd.to_datetime(df.timestamp, unit = 's')

#Initialize Matrix method 1
# matrix = np.zeros((max(df.userId), max(df.movieId)))
# for i in range(df.shape[0]):
#     matrix[df.iloc[i,0]-1, df.iloc[i,1]-1] = df.iloc[i,2]
# #method 2
# mtx = ss.coo_matrix((df.rating, (df.userId, df['movieId'])), shape=(max(df.userId)+1, max(df['movieId'])+1))

#train/validation splinting
trainSet, valSet = sklearn.model_selection.train_test_split(df.iloc[:,:3])
trainSet = ss.coo_matrix((trainSet['rating'], (trainSet['userId'], trainSet['movieId'])), shape=(df['userId'].max()+1,df['movieId'].max()+1))
valSet = ss.coo_matrix((valSet['rating'], (valSet['userId'], valSet['movieId'])), shape=(df['userId'].max()+1,df['movieId'].max()+1))

def validation(W,H, valSet):
    error = 0
    for (row, col, data) in zip(valSet.row, valSet.col, valSet.data):
        error += abs(np.dot(W[row], H[:,col]) - data)
    return error

error = np.zeros(10)
trainloss = np.zeros(10)
for i in range(1,10):
    model = sklearn.decomposition.NMF(n_components=2*i, init = "nndsvd", alpha = 1, max_iter = 1000)
    W = model.fit_transform(trainSet)
    H = model.components_
    trainloss = model.reconstruction_err_
    error[i] = validation(W, H, valSet)

plt.plot([2*i for i in range(1,10)],error[1:] )