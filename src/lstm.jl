struct LSTM
    # TODO
    a
end

rnnparams(lstm::LSTM) = nothing

function rnnconvert(layer::LSTM; atype = Array{Float32})
    params = rnnparams(layer)
    return LSTM(params; atype = atype)
end

function hiddentozero!(layer::LSTM)
    layer.h = zeros(length(layer.bRr))
end

function numberofparameters(layer::LSTM)
    #todo
    return -1
end

function fillparams!(layer::LSTM, params::AbstractVector)
    @assert numberofparameters(layer) == length(params)
    fieldnames = [:a]
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

function show(io::IO, layer::LSTM, depth::Int = 0)
    println("LSTM layer with $(numberofparameters(layer)) parameters")
    #TODO
    return nothing
end
