import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    """
    Perform normalized cuts clustering on a graph.
    
    N-cuts addresses limitations of basic spectral clustering by solving a generalized
    eigenvalue problem. This normalization helps produce more balanced partitions by
    considering the volume (total edge weight) of each partition.
    
    The key difference from spectral clustering is solving Lx = λDx instead of Lx = λx.
    
    Args:
        affinity_mat: Square symmetric numpy array [M*N, M*N] representing the graph
        k: Number of desired clusters (k=2 for recursive case)
    
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

    # Step 3: Solve generalized eigenvalue problem Lx = λDx for k smallest eigenvalues    
    try:
        # Option 1: Direct approach - find k smallest eigenvalues of generalized problem
        # This solves Lx = λDx directly without shifting, getting true smallest eigenvalues
        eigenvalues, eigenvectors = eigsh(L, M=D, k=k, which='SM')
        
        # Option 2: Shift-and-invert with small epsilon (often more numerically stable)
        # eigenvalues, eigenvectors = eigsh(L, M=D, k=k, sigma=1e-10, which='LM')

    except:
        # Fallback: Use dense solver when sparse solver fails to converge
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(L, D)
        eigenvectors = eigenvectors[:, :k]

    # Step 4: Form matrix U with k eigenvectors as columns
    U = eigenvectors

    # Step 5: Apply k-means clustering to row vectors of U
    # Each row of U represents a data point in the spectral embedding
    kmeans = KMeans(n_clusters=k, random_state=1)
    cluster_idx = kmeans.fit_predict(U)
    
    return cluster_idx

def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:
    """
    Calculate the normalized cut metric for binary partitioning.
    
    The n-cut value measures the quality of a partition by considering both the
    strength of connections being cut and the total volume of each partition.
    
    Mathematical Formulation:
    - N_cut(A,B) = 2 - N_assoc(A,B)
    - N_assoc(A,B) = assoc(A,A)/assoc(A,V) + assoc(B,B)/assoc(B,V)
    - assoc(A,V) = Σ_{u∈A,t∈V} W(u,t) (total connection from A to all vertices)
    
    Args:
        affinity_mat: Square symmetric numpy array [M*N, M*N] representing the graph
        cluster_idx: 1D array of length M*N with binary cluster labels (0 or 1)
    
    Returns:
        n_cut_value: The normalized cut value for this partitioning
    """
    W = affinity_mat
    
    # Get indices for each cluster
    cluster_A = np.where(cluster_idx == 0)[0]
    cluster_B = np.where(cluster_idx == 1)[0]
    
    # Handle degenerate cases
    if len(cluster_A) == 0 or len(cluster_B) == 0:
        return 2.0 # If one cluster is empty, return maximal n-cut value


    # Calculate degrees of all nodes
    # Degree of a node is the sum of weights of edges connected to it
    degrees = np.sum(W, axis=1)
    
    # Calculate assoc(A,V) - sum of degrees of nodes in A
    assoc_AV = np.sum(degrees[cluster_A])
    
    # Calculate assoc(B,V) - sum of degrees of nodes in B  
    assoc_BV = np.sum(degrees[cluster_B])

    if assoc_AV < 1e-10 or assoc_BV < 1e-10: 
        # If either partition has no connections, return maximal n-cut value
        return 2.0


    # Calculate assoc(A,A) - sum of weights within cluster A
    assoc_AA = np.sum(W[np.ix_(cluster_A, cluster_A)])
    
    # Calculate assoc(B,B) - sum of weights within cluster B
    assoc_BB = np.sum(W[np.ix_(cluster_B, cluster_B)])
            
    # Calculate N_assoc(A,B) = assoc(A,A)/assoc(A,V) + assoc(B,B)/assoc(B,V)
    N_assoc_AB = (assoc_AA / assoc_AV) + (assoc_BB / assoc_BV)
    
    # Calculate N_cut(A,B) = 2 - N_assoc(A,B)
    # Lower n-cut values indicate better partitions
    n_cut_value = 2.0 - N_assoc_AB
    
    return n_cut_value

def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    """
    Perform recursive normalized cuts segmentation.
    
    This function adaptively determines the number of clusters by recursively
    splitting the graph using binary n-cuts until stopping criteria are met.
    
    Algorithm Logic:
    - Recursively split graph into binary partitions (k=2)
    - Stop splitting when either cluster has fewer than T1 nodes OR N_cut value exceeds T2
    - Continue until no further splits are possible
    - Build a binary tree where each node represents a partition decision
    
    Args:
        affinity_mat: Square symmetric numpy array [M*N, M*N] representing the graph
        T1: Threshold for minimum cluster size (prevents over-segmentation, e.g., 5)
        T2: Threshold for n-cut value (prevents poor cuts, e.g., 0.20)
    
    Returns:
        cluster_idx: 1D numpy array of length M*N with cluster labels
    """
    # Validate inputs
    if not isinstance(affinity_mat, np.ndarray):
        raise TypeError("Affinity matrix must be a numpy array")
    
    if affinity_mat.shape[0] != affinity_mat.shape[1]:
        raise ValueError("Affinity matrix must be square")
    
    if T1 <= 0:
        raise ValueError("T1 must be positive")
    
    if not (0 <= T2 <= 2):
        raise ValueError("T2 must be between 0 and 2 (n-cut value range)")
    
    num_nodes = affinity_mat.shape[0]
    cluster_idx = np.zeros(num_nodes, dtype=int)
    current_label = 0

    # Queue for recursive processing: (indices, affinity_submatrix)
    # Each item represents a subgraph to potentially split further
    queue = [(np.arange(num_nodes), affinity_mat)]
    
    while queue:
        indices, sub_affinity = queue.pop(0)
        
        # Stopping criterion 0: Check if cluster is too small to split
        if len(indices) < 2 * T1:
            cluster_idx[indices] = current_label
            current_label += 1
            continue
        
        # Perform binary n-cuts (k=2) on the subgraph
        sub_clusters = n_cuts(sub_affinity, k=2)
        
        # Get indices for the two resulting clusters
        cluster_0_indices = indices[sub_clusters == 0]
        cluster_1_indices = indices[sub_clusters == 1]
        
        n_cut_val = calculate_n_cut_value(sub_affinity, sub_clusters)
        
        # Check stopping criteria: either cluster too small OR n-cut value too high
        if (len(cluster_0_indices) < T1 or len(cluster_1_indices) < T1 or n_cut_val > T2):
            # Keep the split but don't split these pieces further
            cluster_idx[cluster_0_indices] = current_label
            current_label += 1
            cluster_idx[cluster_1_indices] = current_label
            current_label += 1
        else:
            # Split is acceptable, add both subclusters to queue for further processing
            
            # Extract sub-affinity matrices for each cluster
            sub_affinity_0 = sub_affinity[np.ix_(sub_clusters == 0, sub_clusters == 0)]
            sub_affinity_1 = sub_affinity[np.ix_(sub_clusters == 1, sub_clusters == 1)]
            
            # Add both subclusters to processing queue
            queue.append((cluster_0_indices, sub_affinity_0))
            queue.append((cluster_1_indices, sub_affinity_1))

    return cluster_idx
