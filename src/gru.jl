mutable struct GRU{Tw, Tb}
    Wr::Tw
    Wi::Tw
    Wn::Tw
    Rr::Tw
    Ri::Tw
    Rn::Tw
    bWr::Tb
    bWi::Tb
    bWn::Tb
    bRr::Tb
    bRi::Tb
    bRn::Tb
    h::Union{AbstractArray, KnetArray, UInt8} # Hidden state
    atype::Type
    nX::Int # Number of inputs
    nH::Int # Number of outputs/hidden states
end

# All matrices in here are actually transposed
# Constructor for training
function GRU(nX::Number, nH::Number; atype = Array{Float32})
    Wr = Knet.param(nX, nH; atype = atype)
    Wi = Knet.param(nX, nH; atype = atype)
    Wn = Knet.param(nX, nH; atype = atype)
    Rr = Knet.param(nH, nH; atype = atype)
    Ri = Knet.param(nH, nH; atype = atype)
    Rn = Knet.param(nH, nH; atype = atype)
    bWr = Knet.param0(nH; atype = atype)
    bWi = Knet.param0(nH; atype = atype)
    bWn = Knet.param0(nH; atype = atype)
    bRr = Knet.param0(nH; atype = atype)
    bRi = Knet.param0(nH; atype = atype)
    bRn = Knet.param0(nH; atype = atype)
    h = 0x00
    return GRU(Wr, Wi, Wn, Rr, Ri, Rn, bWr, bWi, bWn, bRr, bRi, bRn, h, atype, nX, nH)
end

function GRU(params::AbstractVector; h = 0x00, atype = Array{Float32})
    @assert(length(params) == 12)
    p2 = similar(params)
    for i = 1:length(params)
        p2[i] = atype(copy(params[i]))
    end
    # TODO check size consistency of the params
    nX = size(p2[1], 1)
    nH = size(p2[4], 1)
    return GRU(p2... , h, atype, nX, nH)
end

# for consistency with Knet
rnnparams(gru::GRU) = [value(gru.Wr), value(gru.Wi), value(gru.Wn), value(gru.Rr), value(gru.Ri), value(gru.Rn), value(gru.bWr), value(gru.bWi), value(gru.bWn), value(gru.bRr), value(gru.bRi), value(gru.bRn)]

function (gru::GRU)(x::Union{AbstractVector, KnetArray{<:Number, 1}})
    # Evaluate the equations for a gru type recurrent neural network
    h = similar(x, gru.nH)
    h .= gru.h
    r = sigm.(gru.Wr' * x .+ gru.Rr' * h .+ gru.bWr .+ gru.bRr) # reset gate
    i = sigm.(gru.Wi' * x .+ gru.Ri' * h .+ gru.bWi .+ gru.bRi) # input gate
    n = tanh.(gru.Wn' * x .+ r .* (gru.Rn' * h .+ gru.bRn) .+ gru.bWn) # new gate
    h2 = (1 .- i) .* n .+ i .* h
    gru.h = h2
    return h2
end


function (gru::GRU)(x::Union{AbstractMatrix, KnetArray{<:Number, 2}})
    # Evaluate the equations for a gru type recurrent neural network
    h = similar(x, gru.nH, size(x, 2))
    h .= gru.h
    r = sigm.(gru.Wr' * x .+ gru.Rr' * h .+ gru.bWr .+ gru.bRr) # reset gate
    i = sigm.(gru.Wi' * x .+ gru.Ri' * h .+ gru.bWi .+ gru.bRi) # input gate
    n = tanh.(gru.Wn' * x .+ r .* (gru.Rn' * h .+ gru.bRn) .+ gru.bWn) # new gate
    h2 = (1 .- i) .* n .+ i .* h
    gru.h = h2
    return h2
end


function (gru::GRU)(X::Union{AbstractArray{<:Number, 3}, KnetArray{<:Number, 3}})
    # If h is not given, set h to zero temporarily
    no_h = (gru.h == 0x00)
    no_h ? gru.h = gru.atype( zeros(size(gru.Rr, 1), size(X, 2)) ) : nothing # set a h temporarily

    # Perform forward pass
    hout = gru.atype(undef, size(gru.Rr, 1), size(X, 2), size(X, 3))
    for t = 1:size(X, 3)
        hout[:, :, t] = gru(X[:, :, t])
    end
    #= Since AutoGrad does not support the upper one, it has to be done like this
    h = Array{Any}(undef, size(X, 3))
    for t = 1:size(X, 3)
        h[t] = gru(X[:, :, t])
    end
    h_dims = (size(gru.Rr, 2), size(X, 2), size(X, 3))
    hout = reshape(hcat(h...), h_dims) # cat(a...; dims = 3) is not implemented in AutoGrad 1.1.3=#

    # Evtntually reset h
    no_h ? (gru.h = 0x00) : nothing
    return hout
end


function rnnconvert(layer::GRU, atype)
    params = rnnparams(layer)
    return GRU(params; atype = atype)
end

function hiddentozero!(layer::GRU)
    layer.h = zeros(length(layer.bRr))
end

function numberofparameters(layer::GRU)
    # TODO maybe derive from nX and nH? But this should be ok also.
    nparams = 0
    nparams += length(layer.Wr)
    nparams += length(layer.Wi)
    nparams += length(layer.Wn)
    nparams += length(layer.Rr)
    nparams += length(layer.Ri)
    nparams += length(layer.Rn)
    nparams += length(layer.bWr)
    nparams += length(layer.bWi)
    nparams += length(layer.bWn)
    nparams += length(layer.bRr)
    nparams += length(layer.bRi)
    nparams += length(layer.bRn)
    return nparams
end