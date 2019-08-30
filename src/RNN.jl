# This module implements recurrent neural networks functionality.
# The implementation uses AutoGrad and Knet functions and uses similar high level syntax
# The main reason this exists if for faster CPU inference in the context of RL
module RNN

using AutoGrad, Knet, Statistics
import Knet.rnnparams
import Base.show

include("rnn_tanh.jl")
export RNN_TANH

# TODO
include("lstm.jl")
export LSTM

include("gru.jl")
export GRU, rnnparams

include("dense.jl")
export Dense

include("batchnorm.jl")
export BatchNorm

include("chain.jl")
export Chain, rnnconvert, fill_batchnorm_stats, hiddentozero!, numberofparameters, rnnparamvec, fillparams!

include("dataslicer.jl")
export DataSlicer

end
