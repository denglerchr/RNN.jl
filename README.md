# RNN.jl

Paket für eine einfache Erstellung und Nutzung von (rekurrenten) neuronalen Netzen, basierend auf Knet. Bei der Erstellung werden generell AutoGrad Parameter erstellt, zum Trainieren. Das Paket beinhaltet allerdings auch Optionen für die Konvertierung zu anderen Datentypen über *rnnconvert*

## Erstellen eines neuronalen Netzes
Layer definieren, hier nur Dense:
  ```julia
  using RNN
  layer1 = Dense(5, 16; activation = tanh)
  layer2 = Dense(16, 2) # activation = identity
  ```

Beide Layer in ein Netz zusammenpacken:

  ```julia
  nn = Chain( (layer1, layer2) )
  ```

Auswerten des neuronalen Netzes (Funktioniert für Vektoren oder auch höherdimensionale Objekte)

  ```julia
  x = randn(5)
  y = nn(x)

  X = randn(5, 10) # 10 Datenpunkte der passenden Größe (jede Spalte 1 Datenpunkte)
  Y = nn(X) # 10 Eingänge auf einmal auswerten
  ```

Konvertieren der *Param{Array{Float32}}* zu *Array{Float64}*

  ```julia
  println( typeof(nn.layers[1].W) )
  nn2 = rnnconvert(nn; atype = Array{Float64})
  println( typeof(nn2.layers[1].W) )
  ```
