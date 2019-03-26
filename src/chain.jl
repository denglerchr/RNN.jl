# Copy from code example in https://github.com/denizyuret/Knet.jl
struct Chain{T}
    layers::T
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)


function rnnconvert(c::Chain; atype = Array{Float32})
    newlayers = Any[]
    for l in c.layers
        nlayer = rnnconvert(l, atype)
        push!(newlayers, nlayer)
    end
    return Chain(Tuple(newlayers))
end

function rnnconvert(layer::Knet.RNN, atype)
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

function hiddentozero!(layer::Knet.RNN)
    layer.h = 0
end
