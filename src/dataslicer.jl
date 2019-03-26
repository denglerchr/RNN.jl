import Base.iterate

# This struct is used to create minibatches, slicing the arrays DataX and DataY along arbitrary dimensions.
# The iterate methods are implemented to iterate over the data, producing minibatches
struct DataSlicer{Tin, Tout}
    #Todo
    DataX::Tin
    DataY::Tin
    outtype::Type
    bindex::Int # Which index of the Data Array to sample from
    minibatch_size::Int # Batch size
    shuffle::Bool # Shuffle minibatches or not TODO not working yet
    indices::Nothing # Contains the indices over which are iterated (will later be used by "shuffle")
end

function DataSlicer(X, Y, minibatch_size; outtype::Type = Array{Float32}, bindex::Int = ndims(X))
    # Check for consistency
    @assert typeof(X) == typeof(Y)
    @assert bindex <= ndims(X)
    @assert size(X, bindex) == size(Y, bindex)
    @assert size(X, bindex) >= minibatch_size
    return DataSlicer{typeof(X), outtype}(X, Y, outtype, bindex, minibatch_size, false, nothing)
end

## Case Input Output of the same type (don't copy data, just return a view)
function iterate(td::DataSlicer{T, T}) where T
    if td.shuffle
        @warn("Shuffling not yet implemented")
    end
    tempX = Any[axes(td.DataX)...]
    tempX[td.bindex] = 1:td.minibatch_size
    outindicesX = CartesianIndices(Tuple(tempX)) # TODO, is there a more efficient way, infering the dimensions from the type?

    tempY = Any[axes(td.DataY)...]
    tempY[td.bindex] = 1:td.minibatch_size
    outindicesY = CartesianIndices(Tuple(tempY)) # TODO, is there a more efficient way, infering the dimensions from the type?

    OutX = view(td.DataX, outindicesX)
    OutY = view(td.DataY, outindicesY)
    return ((OutX, OutY), td.minibatch_size)
end

function iterate(td::DataSlicer{T, T}, state::Int) where T
    # Stop if not enough data to ouput another batch
    if (state+td.minibatch_size)>size(td.DataX, td.bindex)
        return nothing
    end
    tempX = Any[axes(td.DataX)...]
    tempX[td.bindex] = state+1:state+td.minibatch_size
    outindicesX = CartesianIndices(Tuple(tempX)) # TODO, is there a more efficient way, infering the dimensions from the type?

    tempY = Any[axes(td.DataY)...]
    tempY[td.bindex] = state+1:state+td.minibatch_size
    outindicesY = CartesianIndices(Tuple(tempY)) # TODO, is there a more efficient way, infering the dimensions from the type?

    OutX = view(td.DataX, outindicesX)
    OutY = view(td.DataY, outindicesY)
    return ((OutX, OutY), state+td.minibatch_size)
end

## Case Input Output of differen type (copy data into new type)
function iterate(td::DataSlicer{Tin, Tout}) where {Tin, Tout}
    if td.shuffle
        @warn("Shuffling not yet implemented")
    end
    tempX = Any[axes(td.DataX)...]
    tempX[td.bindex] = 1:td.minibatch_size
    outindicesX = CartesianIndices(Tuple(tempX)) # TODO, is there a more efficient way, infering the dimensions from the type?

    tempY = Any[axes(td.DataY)...]
    tempY[td.bindex] = 1:td.minibatch_size
    outindicesY = CartesianIndices(Tuple(tempY)) # TODO, is there a more efficient way, infering the dimensions from the type?

    OutX = td.outtype(td.DataX[outindicesX])
    OutY = td.outtype(td.DataY[outindicesY])
    return ((OutX, OutY), td.minibatch_size)
end

function iterate(td::DataSlicer{Tin, Tout}, state::Int) where {Tin, Tout}
    if (state+td.minibatch_size)>size(td.DataX, td.bindex)
        return nothing
    end
    tempX = Any[axes(td.DataX)...]
    tempX[td.bindex] = state+1:state+td.minibatch_size
    outindicesX = CartesianIndices(Tuple(tempX)) # TODO, is there a more efficient way, infering the dimensions from the type?

    tempY = Any[axes(td.DataY)...]
    tempY[td.bindex] = state+1:state+td.minibatch_size
    outindicesY = CartesianIndices(Tuple(tempY)) # TODO, is there a more efficient way, infering the dimensions from the type?

    OutX = td.outtype(td.DataX[outindicesX])
    OutY = td.outtype(td.DataY[outindicesY])
    return ((OutX, OutY), state+td.minibatch_size)
end