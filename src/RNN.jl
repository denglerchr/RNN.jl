# This module implements recurrent neural networks functionality.
# The implementation uses AutoGrad and Knet functions and uses similar high level syntax
# The main reason this exists if for faster CPU inference in the context of RL
module RNN

using AutoGrad, Knet
import Knet.rnnparams
import Base.show

# TODO
include("lstm.jl")
export LSTM

include("gru.jl")
export GRU, rnnparams

include("dense.jl")
export Dense

include("chain.jl")
export Chain, rnnconvert, hiddentozero!, numberofparameters

include("dataslicer.jl")
export DataSlicer

end
