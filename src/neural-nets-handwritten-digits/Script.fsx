#load "Util.fs"
#load "Matrix.fs"
#load "Network.fs"
#load "Backpropagation.fs"
#load "Training.fs"

open System
open Matrix
open Network
open Backpropagation
open Training

let rnd = Random ()

let startNetwork = createNetwork rnd [784; 30; 10]

let trainingData =
  let image = List.init 784 (fun _ -> 5.0) : Vector
  let digit = List.init 10 (function 3 -> 1.0 | _ -> 0.0)
  List.init 100 (fun _ -> (image, digit))
  |> Array.ofList

printfn "%A" <| stochasticGradientDescent startNetwork trainingData 5 3 3.0 None rnd
