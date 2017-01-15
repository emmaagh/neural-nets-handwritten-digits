module Training

open System
open System.Diagnostics
open Util
open Matrix
open Network
open Backpropagation

let trainNetworkOnMiniBatch nu network miniBatch =
  let miniBatchSize = Array.length miniBatch |> float
  let emptyNetwork = {
    Biases = network.Biases |> List.map vectorOfZeroes;
    Weights = network.Weights |> List.map matrixOfZeroes
  }
  let addNetworks netA netB =
    { Biases = List.map2 (+) netA.Biases netB.Biases;
      Weights = List.map2 (+) netA.Weights netB.Weights }
  let normaliseNablaMatrix (m:Matrix<float>) = m * (nu / miniBatchSize)
  let normaliseNablaVector (v:Vector<float>) = v * (nu / miniBatchSize)
  let nablaNetwork =
    miniBatch
    |> Array.map (backpropagate network)
    |> Array.fold addNetworks emptyNetwork
    |> fun net -> { Biases = net.Biases |> List.map normaliseNablaVector;
                    Weights = net.Weights |> List.map normaliseNablaMatrix }
  { Biases = List.map2 (-) network.Biases nablaNetwork.Biases;
    Weights = List.map2 (-) network.Weights nablaNetwork.Weights }

let countCorrectOutputs network data =
  data
  |> List.filter (fun (x, y) ->
        feedForward network x
        |> fun v -> v.ToArray()
        |> Array.mapi (fun i v -> i, v)
        |> Array.maxBy snd
        |> fst
        |> (=) y)
  |> List.length

let displayEpochResults (stopwatch:Stopwatch) network testData epoch =
  match testData with
  | Some d  -> printfn
                 "Epoch %i: %A / %i (%A)"
                 epoch
                 (countCorrectOutputs network d)
                 (List.length d)
                 stopwatch.Elapsed.TotalSeconds
  | None    -> printfn "Epoch %i complete (%A)" epoch stopwatch.Elapsed.TotalSeconds

let stochasticGradientDescent startNetwork trainingData epochs miniBatchSize nu testData rnd =
  let stopwatch = Stopwatch.StartNew ()
  displayEpochResults stopwatch startNetwork testData -1
  let getMiniBatches () =
    trainingData
    |> shuffle rnd
    |> batchesOf miniBatchSize
    |> Seq.map Array.ofSeq
  let rec trainNetwork network epoch =
    if epoch = epochs then
      network
    else
      let refinedNetwork =
        getMiniBatches ()
        |> Seq.fold (trainNetworkOnMiniBatch nu) network
      displayEpochResults stopwatch refinedNetwork testData epoch
      trainNetwork refinedNetwork <| epoch + 1
  trainNetwork startNetwork 0
