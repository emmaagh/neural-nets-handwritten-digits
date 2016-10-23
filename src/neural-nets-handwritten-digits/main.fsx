#load "Util.fs"
#load "Matrix.fs"
#load "Network.fs"
#load "Backpropagation.fs"
#load "Training.fs"
#load "MnistReader.fs"

open System
open Matrix
open Network
open Backpropagation
open Training
open MnistReader

let train () =
  let rnd = Random ()

  let startNetwork = createNetwork rnd [784; 30; 10]

  printfn "Loading test data"

  let trainingData =
    getTrainingSet ()
    |> Seq.take 1000
    |> Array.ofSeq

  let testData =
    getTestSet ()
    |> Seq.take 50
    |> List.ofSeq
    |> Some

  printfn "Test data loaded"

  let resultNetwork = stochasticGradientDescent startNetwork trainingData 5 100 3.0 testData rnd

  printfn "Biases"
  resultNetwork.Biases
  |> List.iter (List.length >> printfn "%A")

  printfn "Weights"
  resultNetwork.Weights
  |> List.iter (List.length >> printfn "%A")

train ()
