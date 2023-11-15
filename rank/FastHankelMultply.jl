"""
    FastHankelMultply(c,z,false; order="3D")
Fast Hankel matrix-vector product
- c: array(vector) needs to be constructed for Hanke matrix
- z: vector
- adj: Forward or Adjoint
- order: for 2D, 3D or 5D data constructing Hankel matrix
# References
* Cheng et. al. 2019. Computational efficient multidimensional singular spectrum
analysis for prestack seismic data reconstruction, Geophysics, 84, 2, V111–V119.
Author: Rongzhi Lin (SAIG in UofA)
"""
function FastHankelMultply(c,v,adj::Bool; order="3D")

    if order == "2D"

        N = length(c);
        K = Int64(floor(N/2)) + 1;
        L = N + 1 - K;

        if adj == false
            v    = copy(reverse(v));
            vhat = [v;zeros(L-1)];
            r    = ifft(fft(c) .* fft(vhat));
            r    = r[K:N];
        else
            v    = copy(reverse(v));
            vhat = [v;zeros(K-1)];
            r    = ifft(fft(conj(c)) .* fft(vhat));
            r    = r[L:N];
        end

    #### for 3D data, but operate in 2D dimention
    elseif order == "3D"
        (N1,N2) = size(c)
        K1      = Int64(floor(N1/2) ) + 1;
        L1      = N1 + 1 - K1;

        K2      = Int64(floor(N2/2) ) + 1;
        L2      = N2 + 1 - K2;

        if adj == false
            v1    = copy(reverse(reverse(v, dims=1), dims=2))
            padv1 = zeros(eltype(v1),L1-1,L2-1)
            vhat  = cat(v1,padv1;dims=(1,2));

            r     = ifft(fft(c) .* fft(vhat));
            r     = r[K1:N1,K2:N2];
            r     = copy(reshape(r,size(r,1)*size(r,2)))

        else

            v1    = copy(reverse(reverse(v, dims=1), dims=2))
            padv1 = zeros(eltype(v1),K1-1,K2-1)
            vhat  = cat(v1,padv1;dims=(1,2));
            r   = ifft(fft(conj(c)) .* fft(vhat));
            r   = r[L1:N1,L2:N2];
            r   = copy(reshape(r,size(r,1)*size(r,2)))
        end

        #### for 5D data, but operate in 4D dimention
    elseif order == "5D"

        (N1,N2,N3,N4) = size(c)
        #### Size of Hankel MAtrix in x.
        K1 = Int64(floor(N1/2) ) + 1;
        L1 = N1 + 1 - K1;

        #### Size of Hankel MAtrix in y.
        K2 = Int64(floor(N2/2) ) + 1;
        L2 = N2 + 1 - K2;

        #### Size of Hankel MAtrix in Z.
        K3 = Int64(floor(N3/2) ) + 1;
        L3 = N3 + 1 - K3;

        #### Size of Hankel MAtrix in W.
        K4 = Int64(floor(N4/2) ) + 1;
        L4 = N4 + 1 - K4;

        #### reverse vector v
        v1 = reverse(v, dims=1)
        v2 = reverse(v1, dims=2)
        v3 = reverse(v2, dims=3)
        v4 = reverse(v3, dims=4)

        #### Calculate H*m
        if adj == false

            padv4 = zeros(eltype(v4),L1-1,L2-1,L3-1,L4-1)
            vhat = cat(v4,padv4;dims=(1,2,3,4));
            r = ifft(fft(c) .* fft(vhat));
            r = r[K1:N1,K2:N2,K3:N3,K4:N4];
            r = copy(reshape(r,size(r,1)*size(r,2)*size(r,3)*size(r,4)))
        else

            padv4 = zeros(eltype(v4),K1-1,K2-1,K3-1,K4-1)
            vhat = cat(v4,padv4;dims=(1,2,3,4));
            r   = ifft(fft(conj(c)) .* fft(vhat));
            r = r[L1:N1,L2:N2,L3:N3,L4:N4];
            r = copy(reshape(r,size(r,1)*size(r,2)*size(r,3)*size(r,4)))
        end

    else
        error("Order should be equal to \"2D\" or \"3D\" or  \"5D\" ")
    end



    return r
end
