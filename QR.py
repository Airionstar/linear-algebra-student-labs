import numpy as np

def gram_schmidt_qr(A):
    """
    Compute the QR factorisation of a square matrix using the classical
    Gram-Schmidt process.

    Parameters
    ----------
    A : numpy.ndarray
        A square 2D NumPy array of shape ``(n, n)`` representing the input
        matrix.

    Returns
    -------
    Q : numpy.ndarray
        Orthonormal matrix of shape ``(n, n)`` where the columns form an
        orthonormal basis for the column space of A.
    R : numpy.ndarray
        Upper triangular matrix of shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"the matrix A is not square, {A.shape=}")

    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        # Start with the j-th column of A
        u = A[:, j].copy()

        # Orthogonalize against previous q vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # projection coefficient
            u -= R[i, j] * Q[:, i]  # subtract the projection

        # Normalize u to get q_j
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]

    return Q, R

def errors(A, Q, R):
    error1 = np.linalg.norm(A - np.dot(Q,R))
    QTranspose = Q.transpose()
    error2 = np.linalg.norm(np.dot(QTranspose, Q) - np.eye(2))
    RUpper = np.triu(R)
    error3 = np.linalg.norm(R - RUpper)

    return error1, error2, error3

table = []
for i in range(6,17):
    A = np.array([[1, 1+10**-i], [1+10**-1, 1]])
    Q, R = gram_schmidt_qr(A)
    error1, error2, error3 = errors(A, Q, R)
    table.append([10**-i, error1, error2, error3])
table = np.array(table)
print("E\t\tError1\t\tError2\t\tError3")
for row in table:
    e = row[0]
    error1 = row[1]
    error2 = row[2]
    error3 = row[3]

    formattede = f"{e:.2e}"
    formattedError1 = f"{error1:.2e}"
    formattedError2 = f"{error2:.2e}"
    formattedError3 = f"{error3:.2e}"

    print(f"{formattede}\t{formattedError1}\t{formattedError2}\t{formattedError3}")