# Copy from code example in https://github.com/denizyuret/Knet.jl
struct Chain{T}
    layers::T
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

"""
Change array type of every layer.
"""
function rnnconvert(c::Chain; atype = Array{Float32})
    newlayers = Any[]
    for l in c.layers
        nlayer = rnnconvert(l; atype = atype)
        push!(newlayers, nlayer)
    end
    return Chain(Tuple(newlayers))
end

function rnnconvert(layer::Knet.RNN; atype = Array{Float32})
    params = Knet.rnnparams(layer)
    if layer.mode == 1
        params[3] .+= params[4] # Add both biases together
        return RNN_TANH(params[1:3], atype = atype)
    elseif layer.mode == 2
        return LSTM(params; atype = atype)
    elseif layer.mode == 3
        return GRU(params; atype = atype)
    else
        error("rnn type not supported")
    end
end

"""
Use a batch of data to create statistics for your layer inputs.
Converts all batchnorm layers to be used for inference.
"""
function fill_batchnorm_stats(c::Chain, X)
    # Check if there is a BatchNorm layer (is there an or function operator over arrays?)
    temp = isa.(BatchNorm{false}, c.layers)
    hasBN = !prod(.!(temp))

    # Return current Chain if there is no BatchNorm layer
    !hasBN && return c

    # Else convert BatchNorm layer
    layers = Array{Any}(undef, length(c.layers))
    for (i, l) in enumerate(c.layers)
        if isa(BatchNorm{false}, l)
            layers[i] = BatchNorm{true}(X)
        else
            layers[i] = l
        end
        X = l(X)
    end
    return Chain(Tuple(layers))
end


function hiddentozero!(c::Chain)
    for l in c.layers
        hiddentozero!(l)
    end
    return nothing
end

function numberofparameters(c::Chain)
    nparams = 0
    for l in c.layers
        nparams += numberofparameters(l)
    end
    return nparams
end

function hiddentozero!(layer::Knet.RNN)
    layer.h = 0
end

function numberofparameters(layer::Knet.RNN)
    return length(layer.w)
end

"""
Return all the parameters in a Chain or layer as a vector
"""
function rnnparamvec(c::Chain; dtype::DataType = Float32)
    Out = Array{dtype}(undef, numberofparameters(c))
    index = 1 # running index, where to write params of current layer
    for l in c.layers
        n = numberofparameters(l)
        Out[index:index+n-1] .= rnnparamvec(l; dtype = dtype)
        index += n
    end
    @assert index == length(Out)+1
    return Out
end

function fillparams!(c::Chain, params::AbstractVector)
    @assert numberofparameters(c) == length(params)
    index = 1
    for l in c.layers
        n = numberofparameters(l)
        fillparams!(l, view(params, index:index+n-1))
        index += n
    end
    @assert index == length(params) + 1
    return c
end

function show(io::IO, c::Chain, depth::Int = 0)
    println("NN chain with $(numberofparameters(c)) parameters. Layers:")
    for l in c.layers
        show(io, l)
    end
    return nothing
end
