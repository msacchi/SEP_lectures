function rqrd(in_mat, p)
    # Applying randomized QR for rank reduction

    # Get the number of columns
    ny = size(in_mat, 2)

    # Generate a random matrix
    omega = randn(ny, p)

    # Compute the product
    y = in_mat * omega

    # QR decomposition
    F = qr(y)
    Q=F.Q

    # Calculate the output
    out = Q[:,1:p] * Q[:,1:p]' * in_mat

    return out
end


