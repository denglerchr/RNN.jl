mutable struct RNN_TANH{Tw, Tb}
    Wx::Tw
    Wh::Tw
    b::Tb
    h::Union{AbstractArray, KnetArray, UInt8} # Hidden state
    atype::Type
    nX::Int # Number of inputs
    nH::Int # Number of outputs/hidden states
end

# All matrices in here are actually transposed
# Constructor for training
function RNN_TANH(nX::Number, nH::Number; atype = Array{Float32})
    Wx = Knet.param(nX, nH; atype = atype)
    Wh = Knet.param(nH, nH; atype = atype)
    b = Knet.param0(nH; atype = atype)
    h = 0x00
    return RNN_TANH(Wx, Wh, b, h, atype, nX, nH)
end

function RNN_TANH(params::AbstractVector; h = 0x00, atype = Array{Float32})
    @assert(length(params) == 3)
    p2 = Vector{Any}(undef, length(params))
    for i = 1:length(params)
        p2[i] = atype(copy(params[i]))
    end
    # TODO check size consistency of the params
    nX = size(p2[1], 1)
    nH = size(p2[2], 1)
    return RNN_TANH(p2... , h, eltype([Knet.value(p) for p in p2]), nX, nH)
end

# for consistency with Knet
rnnparams(rnn::RNN_TANH) = [value(rnn.Wx), value(rnn.Wh), value(rnn.b)]

function (rnn::RNN_TANH)(x::Union{AbstractVector, KnetArray{<:Number, 1}})
    h = similar(x, rnn.nH)
    h .= rnn.h
    h2 = tanh.( rnn.Wx' * x .+ rnn.Wh' * h .+ rnn.b )
    rnn.h = h2
    return h2
end


function (rnn::RNN_TANH)(x::Union{AbstractMatrix, KnetArray{<:Number, 2}})
    h = similar(x, rnn.nH, size(x, 2))
    h .= rnn.h
    h2 = tanh.( rnn.Wx' * x .+ rnn.Wh' * h .+ rnn.b )
    rnn.h = h2
    return h2
end


function (rnn::RNN_TANH)(X::Union{AbstractArray{<:Number, 3}, KnetArray{<:Number, 3}})
    # If h is not given, set h to zero temporarily
    no_h = (rnn.h == 0x00)
    no_h ? rnn.h = rnn.atype( zeros(eltype(rnn.atype), rnn.nH, size(X, 2)) ) : nothing # set a h temporarily

    # Perform forward pass
    hout = rnn.atype(undef, rnn.nH, size(X, 2), size(X, 3))
    for t = 1:size(X, 3)
        hout[:, :, t] = rnn(X[:, :, t])
    end

    # Eventually reset h
    no_h ? (rnn.h = 0x00) : nothing
    return hout
end


function rnnconvert(layer::RNN_TANH; atype = Array{Float32})
    params = rnnparams(layer)
    return RNN_TANH(params; atype = atype)
end

function hiddentozero!(layer::RNN_TANH)
    layer.h = 0x00
end

function numberofparameters(layer::RNN_TANH)
    nparams = 0
    nparams += length(layer.Wx)
    nparams += length(layer.Wh)
    nparams += length(layer.b)
    return nparams
end

function rnnparamvec(layer::RNN_TANH; dtype = Float32)
    return Array{dtype}(vcat(layer.Wx[:], layer.Wh[:], layer.b[:]))
end

function fillparams!(layer::RNN_TANH, params::AbstractVector)
    @assert numberofparameters(layer) == length(params)
    fieldnames = [:Wx, :Wh, :b]
    index = 1
    for name in fieldnames
        W = getfield(layer, name)
        n = prod(size(W))
        vec(W) .= params[index:index+n-1]
        index += n
    end
    @assert index == length(params) + 1 # can be deleted, just used to make sure at first run
    return layer
end

function show(io::IO, layer::RNN_TANH, depth::Int = 0)
    println("RNN_TANH layer with $(numberofparameters(layer)) parameters")
    println("\t$(layer.nX) Inputs")
    println("\t$(layer.nH) Outputs/Hidden state size")
    return nothing
end
