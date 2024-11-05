import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

class NMFRecommender:

    def __init__(self,random_state=15,rank=2,maxiter=200,tol=1e-3):
        """
        Save the parameter values as attributes.
        """
        self.random_state = random_state
        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol
  

    def _initialize_matrices(self, m, n):
        """
        Initialize the W and H matrices.
        
        Parameters:
            m (int): the number of rows
            n (int): the number of columns
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """

        # set the random seed
        np.random.seed(self.random_state)

        # initialize the matrices
        W = np.random.random((m,self.rank))
        H = np.random.random((self.rank,n))
        return W, H


    def _compute_loss(self, V, W, H):
        """
        Compute the loss of the algorithm according to the 
        Frobenius norm.
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        """
        return np.linalg.norm(V - W @ H, 'fro')


    def _update_matrices(self, V, W, H):
        """
        The multiplicative update step to update W and H
        Return the new W and H (in that order).
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        Returns:
            New W ((m,k) array)
            New H ((k,n) array)
        """

        # update the matrices
        H = H * (W.T @ V) / (W.T @ W @ H)
        W = W * (V @ H.T) / (W @ H @ H.T)
        return W, H


    def fit(self, V):
        """
        Fits W and H weight matrices according to the multiplicative 
        update algorithm. Save W and H as attributes and return them.
        
        Parameters:
            V ((m,n) array): the array to decompose
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """

        # initialize the matrices
        W, H = self._initialize_matrices(V.shape[0], V.shape[1])
        for _ in range(self.maxiter):

            # update the matrices
            W, H = self._update_matrices(V, W, H)

            # check if the loss is less than the tolerance
            if self._compute_loss(V, W, H) < self.tol:
                break
        self.W = W
        self.H = H
        return W, H


    def reconstruct(self):
        """
        Reconstruct and return the decomposed V matrix for comparison against 
        the original V matrix. Use the W and H saved as attrubutes.
        
        Returns:
            V ((m,n) array): the reconstruced version of the original data
        """
        return self.W @ self.H


def prob4(rank=2):
    """
    Run NMF recommender on the grocery store example.
    
    Returns:
        W ((m,k) array)
        H ((k,n) array)
        The number of people with higher component 2 than component 1 scores
    """
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])
                  
    # run the NMF recommender
    nmf = NMFRecommender(rank=rank)
    W, H = nmf.fit(V)

    # count the number of people with higher component 2 than component 1 scores
    return W, H, np.sum(H[1] > H[0])


def prob5(filename='artist_user.csv'):
    """
    Read in the file `artist_user.csv` as a Pandas dataframe. Find the optimal
    value to use as the rank as described in the lab pdf. Return the rank and the reconstructed matrix V.
    
    Returns:
        rank (int): the optimal rank
        V ((m,n) array): the reconstructed version of the data
    """

    # read in the data
    df = pd.read_csv(filename, index_col=0)
    
    # calculate the frobenius norm of the original matrix
    X = df.values

    # set the benchmark
    benchmark = np.linalg.norm(X, 'fro') * .0001

    # find the optimal rank
    for rank in range(10, 16):
        model = NMF(n_components=rank, init='random', random_state=0)

        # fit the model
        W, H = model.fit_transform(X), model.components_

        # reconstruct the matrix
        V = W @ H

        # calculate the rmse
        rmse = np.sqrt(mse(X, V))

        # check if the rmse is less than the benchmark
        if rmse < benchmark:
            break
    return rank, V


def discover_weekly(userid, V):
    """
    Create the recommended weekly 30 list for a given user.
    
    Parameters:
        userid (int): which user to do the process for
        V ((m,n) array): the reconstructed array
        
    Returns:
        recom (list): a list of strings that contains the names of the recommended artists
    """
    
    # read in the data to get the artist names
    artists = pd.read_csv('artists.csv')['name'].values

    # load artist_user.csv, get ids
    df = pd.read_csv('artist_user.csv', index_col=0)
    userid_to_idx = {id: idx for idx, id in enumerate(df.index)}

    user_idx = userid_to_idx[userid]

     # Get the user's listening data and create the mask for artists not listened to
    user_data = df.loc[userid]
    mask = user_data.values == 0  # True for artists the user hasn't listened to
    
    # Get the user's preferences from the reconstructed matrix using the user index
    user_preferences = V[user_idx, :]
    
    # Filter the preferences to only include those artists the user has not listened to
    masked_preferences = user_preferences[mask]
    
    # Get the indices of the preferences after masking
    masked_indices = np.arange(len(user_preferences))[mask]
    
    # Sort the masked preferences in descending order and get the sorted indices
    sorted_indices = np.argsort(-masked_preferences)
    
    # Use the sorted indices to index into the masked_indices to get the final order
    recommended_indices = masked_indices[sorted_indices]
    
    # Get the top 30 artist names from the masked, sorted indices
    recom = artists[recommended_indices][:30].tolist()
    
    return recom
    
    
if __name__ == '__main__':
    # run the NMF recommender on the grocery store example
    print(prob4())
    
    # find the optimal rank for the artist_user.csv data
    print(prob5())
    
    # create the recommended weekly 30 list for user 2
    print(discover_weekly(2, prob5()[1]))