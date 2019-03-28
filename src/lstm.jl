struct LSTM
    # TODO
    a
end

rnnparams(lstm::LSTM) = nothing

function rnnconvert(layer::LSTM, atype)
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