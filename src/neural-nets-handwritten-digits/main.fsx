#load "Util.fs"
#load "Matrix.fs"
#load "Network.fs"
#load "Backpropagation.fs"
#load "Training.fs"
#load "MnistReader.fs"

open System
open System.Diagnostics
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
    |> Array.ofSeq
    |> Array.map (
        fun (input, output) ->
          input |> Array.ofList |> Vector,
          output |> Array.ofList |> Vector)

  let testData =
    getTestSet ()
    |> List.ofSeq
    |> List.map (fun (input, output) ->
        input |> Array.ofList |> Vector,
        output)
    |> Some

  printfn "Test data loaded"

  let stopwatch = Stopwatch.StartNew ()
  printfn "START"

  stochasticGradientDescent startNetwork trainingData 5 10 3.0 testData rnd
  |> ignore

  printfn "END %f" stopwatch.Elapsed.TotalSeconds

train ()
