function SSA(DATA, dt, P, flow, fhigh, meth)
    # SSA for FX denosising. Explicit Hankel matrices are used in this method
    # input is data(t,x) which is transform to data(f,x) and 1D ssa is applied to
    # each frequency f. Follows Oropeza and Sacchi, 2010 (Geophysics)
    
    nt, nx = size(DATA)
    nf = 4 * nextpow(2,nt)

    DATA_FX_f = zeros(Complex{Float64}, nf, nx)

    # First and last samples of the DFT.
    ilow = max(floor(Int, flow * dt * nf) + 1, 1)
    ihigh = min(floor(Int, fhigh * dt * nf) + 1, floor(Int,nf / 2) + 1)

    # Transform to FX
    DATA_FX_tmp = fft(vcat(DATA,zeros(nf-nt,nx)),1)

    # Size of Hankel Matrix
    Lcol = floor(Int, nx / 2) + 1
    Lrow = nx - Lcol + 1
   

    # Form level-1 block Hankel matrix
    for j in ilow:ihigh
        M = zeros(Complex{Float64}, Lrow, Lcol)

        for lc in 1:Lcol
            M[:, lc] = DATA_FX_tmp[j, lc:lc + Lrow - 1]
        end

        # SVD decomposition with P largest singular values or Randomized SVD
         if meth == "svd"
                    F = svd(M)
                    U = (F.U[:,1:P])
                Mout = U*U'*M
         end
          if meth == "rqrd"
                  Mout = rqrd(M, P) # Implementation of `rand_svd` is required
          end
                  

        # Sum along anti-diagonals to recover signal
        Count = zeros(nx)
        tmp2 = zeros(Complex{Float64}, nx)

        for ic in 1:Lcol
            for ir in 1:Lrow
                Count[ir + ic - 1] += 1
                tmp2[ir + ic - 1] += Mout[ir, ic]
            end
        end

        tmp2 .= tmp2 ./ Count
        DATA_FX_f[j, :] = tmp2
    end

    # Honor symmetries
    for k in floor(Int,nf/2)+2:nf
        DATA_FX_f[k, :] = conj(DATA_FX_f[nf - k + 2, :])
    end

    # Back to TX (the output)
    DATA_f = real(ifft(DATA_FX_f, 1))
    DATA_f = DATA_f[1:nt, :]

    return DATA_f
end

