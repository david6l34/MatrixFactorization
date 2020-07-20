import pandas as pd
import numpy as np
from scipy.sparse import rand as sprand
import torch
import scipy
from sklearn.model_selection import train_test_split

# Make up some random explicit feedback ratings
# and convert to a numpy array
# n_users = 1000
# n_items = 1000
# ratings = sprand(n_users, n_items,
#                  density=0.01, format='csr')
# ratings.data = (np.random.randint(1, 5,
#                                   size=ratings.nnz)
#                           .astype(np.float64))
# ratings = ratings.toarray()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    # ratings = ratings.iloc[:,:3].to_numpy()
    # ratings[:,:2] -= 1
    #rows, cols = ratings[:,0].astype(int), ratings[:, 1].astype(int)
    # train, test = train_test_split(ratings)
    ratings = scipy.sparse.coo_matrix((ratings.iloc[:,2],(ratings.iloc[:,0].astype(int),ratings.iloc[:,1].astype(int))))
    ratings = ratings.toarray()
    rows, cols = ratings.nonzero()

    class MatrixFactorization(torch.nn.Module):

        def __init__(self, n_users, n_items, n_factors=20):
            super().__init__()
            self.user_factors = torch.nn.Embedding(n_users,
                                                   n_factors,
                                                   sparse=True)
            self.item_factors = torch.nn.Embedding(n_items,
                                                   n_factors,
                                                   sparse=True)

        def forward(self, user, item):
            return (self.user_factors(user) * self.item_factors(item)).sum(1)

    # class MovieDataset(torch.utils.data.Dataset):
    #
    #     def __init__(self, data):
    #         # self.userId = torch.from_numpy(data[:,0])
    #         # self.movieId = torch.from_numpy(data[:,1])
    #         # self.rating = torch.from_numpy(data[:,2])
    #
    #     def __getitem__(self, row, col):
    #         return

    model = MatrixFactorization(rows.max()+1, cols.max()+1, n_factors=20).cuda()
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-1) # learning rate

    p = np.random.permutation(len(rows))
    rows, cols = rows[p], cols[p]
    fraction = [int(rows.shape[0]*4/5), int(rows.shape[0] - int(rows.shape[0]*4/5))]

    rows, cols = torch.LongTensor(rows), torch.LongTensor(cols)
    ratings_tensor = torch.FloatTensor(ratings).cuda()
    train_rows, test_rows= torch.split(rows, fraction)
    train_cols, test_cols = torch.split(cols, fraction)


    dataset_train = torch.utils.data.TensorDataset(train_rows, train_cols)
    dataset_test = torch.utils.data.TensorDataset(test_rows, test_cols)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, num_workers=8, batch_size=48, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, num_workers=8, batch_size=48, shuffle=True)

    # Sort our data

    loss_log = []
    loss_log_test = []
    for _ in range(10):
        for row, col in dataloader_train:
            # Turn data into tensors
            rating = ratings_tensor[row, col]
        #     row = torch.LongTensor([row])
        #     col = torch.LongTensor([col])
    #         print(row, col)
            row = row.cuda()
            col = col.cuda()

            # Predict and calculate loss
            prediction = model(row, col)
            loss = loss_func(prediction, rating)

            # Backpropagate
            loss.backward()

            # Update the parameters
            optimizer.step()
            loss_log.append(loss)
            optimizer.zero_grad()

        for row, col in dataloader_test:
            # Turn data into tensors
            rating = ratings_tensor[row, col]
        #     row = torch.LongTensor([row])
        #     col = torch.LongTensor([col])
    #         print(row, col)
            row = row.cuda()
            col = col.cuda()

            # Predict and calculate loss
            prediction = model(row, col)
            loss = loss_func(prediction, rating)

            loss_log_test.append(loss)

    import matplotlib.pyplot as plt


    fig, ax = plt.subplots(2)
    fig.suptitle('10 iteration , 2048 batch size')
    ax[0].plot(loss_log)
    ax[0].set_title('train loss')
    ax[0].set_yscale('log')
    ax[1].plot(loss_log_test)
    ax[1].set_title('val loss')
    ax[1].set_yscale('log')




