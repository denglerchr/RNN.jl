""" Implement batch normalisation.

**BatchNorm{Inference, T<:Union{AbstractVector, KnetArray, Nothing}}**

Inference should be false for training and true for inference. Create for inference using a batch
`X` and calling `BatchNorm{true}(X) where {T}` and for training
calling `BatchNorm()`"""
struct BatchNorm{Inference, T<:Union{AbstractVector, KnetArray, Nothing}}
    m::T # mean
    u::T # standard variance

    function BatchNorm(m, u, inference::Val{true})
        @assert typeof(m) == typeof(u)
        @assert length(m) == length(u)
        return new{true, typeof(m)}(m, u)
    end
    function BatchNorm(inference::Val{false})
        return new{false, Nothing}(nothing, nothing)
    end

end

# Create an inference BatchNorm layer from a Dataset
function BatchNorm{true}(X) where {T}
    m = vec(mean(X, dims = 2:ndims(X)))
    u = vec(std(X, dims = 2:ndims(X)))
    return BatchNorm(m, u, Val(true))
end

# Create a batchnorm for training
function BatchNorm{false}()
    return BatchNorm(Val(false))
end

BatchNorm() = BatchNorm{false}()

# Used for training
function (l::BatchNorm{false})(X)
    m = mean(X, dims = 2)
    u = std(X, dims = 2)
    return  (X .- m)./u
end

# Used for inference
(l::BatchNorm{true})(X) = (X .- l.m)./l.u

function rnnconvert(layer::BatchNorm{true}; atype = Array{Float32})
    m = atype(value(layer.m))
    u = atype(value(layer.u))
    return BatchNorm(m, u, Val(true))
end

rnnconvert(layer::BatchNorm{false}; atype = Array{Float32}) = BatchNorm(nothing, nothing, Val(false))

hiddentozero!(layer::BatchNorm) = nothing

function numberofparameters(layer::BatchNorm{true})
    return length(layer.m) + length(layer.u)
end

function numberofparameters(layer::BatchNorm{false})
    return 0
end

function rnnparamvec(layer::BatchNorm{true}; dtype = Float32)
    return Array{dtype}(vcat(layer.m, layer.u))
end

function fillparams!(layer::BatchNorm{true}, params::AbstractVector)
    @assert numberofparameters(layer) == length(params)
    fieldnames = [:m, :u]
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

function fillparams!(layer::BatchNorm{false}, params::AbstractVector)
    @assert length(params) == 0
    return nothing
end

function show(io::IO, layer::BatchNorm{true}, depth::Int = 0)
    println("BatchNorm layer containing $(numberofparameters(layer)) parameters, used for inference")
    return nothing
end

function show(io::IO, layer::BatchNorm{false}, depth::Int = 0)
    println("BatchNorm layer without fixed parameters, used for training")
    return nothing
end
