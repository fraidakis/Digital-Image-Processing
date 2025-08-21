import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    """
    Perform spectral clustering on a graph represented by an affinity matrix.
    
    Spectral clustering leverages the eigenstructure of the graph Laplacian to find
    natural partitions. The algorithm transforms the clustering problem into a 
    lower-dimensional space where clusters become more apparent.
    
    Algorithm Steps:
    1. Use input affinity matrix as undirected graph W
    2. Calculate Laplacian matrix L = D - W where D(i,i) = Σ_j W(i,j)
    3. Solve eigenvalue problem Lx = λx for k smallest eigenvalues
    4. Form matrix U with k eigenvectors as columns
    5. Apply k-means clustering to row vectors of U
    
    Args:
        affinity_mat: Square symmetric numpy array [M*N, M*N] representing the graph
        k: Number of desired clusters
    
    Returns:
        cluster_idx: 1D numpy array of length M*N with cluster labels
    """
    # Validate inputs
    if not isinstance(affinity_mat, np.ndarray):
        raise TypeError("Affinity matrix must be a numpy array")
    
    if affinity_mat.shape[0] != affinity_mat.shape[1]:
        raise ValueError("Affinity matrix must be square")
    
    if k <= 0 or k > affinity_mat.shape[0]:
        raise ValueError("k must be positive and not exceed matrix size")
    
    # Step 1: Input is already the affinity matrix W
    W = affinity_mat
    n = W.shape[0]
    
    # Step 2: Calculate degree matrix D and Laplacian matrix L = D - W
    # D is diagonal with D(i,i) = sum of i-th row of W (total connection strength)
    degree_vec = np.sum(W, axis=1)
    D = np.diag(degree_vec)
    L = D - W
    
    # Step 3: Solve eigenvalue problem Lx = λx for k smallest eigenvalues
    # Use scipy.sparse.linalg.eigsh for efficient computation
    try:
        # Option 1: Direct approach - find k smallest eigenvalues of standard problem
        # SM = "Smallest Magnitude" - finds eigenvalues with smallest absolute values
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

        # Option 2: Shift-and-invert with small epsilon (often more numerically stable)
        # This finds eigenvalues closest to 1e-10, which are effectively the smallest ones
        # The shift helps avoid numerical issues when eigenvalues are very close to zero
        # eigenvalues, eigenvectors = eigsh(L, k=k, sigma=1e-10, which='LM')

    except:
        # Fallback: Use dense solver when sparse solver fails to converge
        # scipy.linalg.eigh handles the full standard eigenvalue problem
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        eigenvectors = eigenvectors[:, :k]

    # Step 4: Form matrix U with k eigenvectors as columns
    U = eigenvectors
    
    # Step 5: Apply k-means clustering to row vectors of U
    # Each row of U represents a data point in the spectral embedding
    kmeans = KMeans(n_clusters=k, random_state=1)
    cluster_idx = kmeans.fit_predict(U)
    
    return cluster_idx