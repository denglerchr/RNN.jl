# RNN.jl

Paket für eine einfache Erstellung und Nutzung von (rekurrenten) neuronalen Netzen, basierend auf Knet. Bei der Erstellung werden generell AutoGrad Parameter erstellt, zum Trainieren. Das Paket beinhaltet allerdings auch Optionen für die Konvertierung zu anderen Datentypen über *rnnconvert*

## Erstellen eines neuronalen Netzes
Layer definieren, hier nur Dense:

  using RNN
  layer1 = Dense(5, 16; activation = tanh)
  layer2 = Dense(16, 2) # activation = identity

Beide Layer in ein Netz zusammenpacken:

  nn = Chain( (layer1, layer2) )

Auswerten des neuronalen Netzes

  x = randn(5)
  y = nn(x)

Konvertieren der *Param{Array{Float32}}* zu *Array{Float64}*

  nn2 = rnnconvert(nn; atype = Array{Float64})
