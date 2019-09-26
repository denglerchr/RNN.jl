# Fully connectes layer, for use after an RNN layer
struct Dense{Tw, Tb}
    W::Tw # Weights
    b::Tb # Bias
    f::Function # Activation function
    atype::Type
end

# Evaluate the layer in context of RNN
function (d::Dense)(X)
    sizeX2 = size(X)[2:end]
    Out2D = d.f.(d.W * mat(X, dims=1) .+ d.b)
    return  reshape(Out2D, (size(d.W, 1), sizeX2...) )
end

function (d::Dense)(X::Union{AbstractMatrix, KnetArray{T, 2}}) where {T}
    sizeX2 = size(X, 2)
    Out2D = d.f.(d.W * X .+ d.b)
    return  reshape(Out2D, (size(d.W, 1), sizeX2) )
end

function (d::Dense)(x::T) where {T<:AbstractVector}
    Out = similar(x, length(d.b))
    Out .= d.f.(d.W * x .+ d.b)
    return Out::T
end

function rnnconvert(layer::Dense; atype = Array{Float32})
    newW = atype(value(layer.W))
    newb = atype(value(layer.b))
    return Dense(newW, newb, layer.f, typeof(Knet.value(newW))
end

Dense(nIn::Int, nOut::Int; atype = Array{Float32}, activation = identity) = Dense(Knet.param(nOut, nIn; atype = atype), Knet.param0(nOut; atype = atype), activation, atype)

hiddentozero!(layer::Dense) = nothing

function numberofparameters(layer::Dense)
    return length(layer.W) + length(layer.b)
end

function rnnparamvec(layer::Dense; dtype = Float32)
    return Array{dtype}(vcat(layer.W[:], layer.b[:]))
end

function fillparams!(layer::Dense, params::AbstractVector)
    @assert numberofparameters(layer) == length(params)
    fieldnames = [:W, :b]
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

function show(io::IO, layer::Dense, depth::Int = 0)
    println("Fully Connected layer with $(numberofparameters(layer)) parameters")
    println("\t$(size(layer.W, 2)) Inputs")
    println("\t$(size(layer.W, 1)) Outputs")
    return nothing
end
