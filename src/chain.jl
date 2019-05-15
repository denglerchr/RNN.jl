# Copy from code example in https://github.com/denizyuret/Knet.jl
struct Chain{T}
    layers::T
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)


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
    if layer.mode == 2
        return LSTM(params; atype = atype)
    elseif layer.mode == 3
        return GRU(params; atype = atype)
    else
        error("rnn type not supported")
    end
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

function show(io::IO, c::Chain, depth::Int = 0)
    println("NN chain with $(numberofparameters(c)) parameters. Layers:")
    for l in c.layers
        show(io, l)
    end
    return nothing
end
