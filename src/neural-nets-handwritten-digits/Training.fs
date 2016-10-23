module Training

open System
open Util
open Matrix
open Network
open Backpropagation

let trainNetworkOnMiniBatch nu network miniBatch =
  let miniBatchSize = List.length miniBatch |> double
  let always x _ = x
  let emptyNetwork = {
    Biases = network.Biases |> List.map (List.map (always 0.0));
    Weights = network.Weights |> List.map (List.map (List.map (always 0.0)))
  }
  let addNetworks netA netB =
    { Biases = List.map2 add netA.Biases netB.Biases;
      Weights = List.map2 addMatrices netA.Weights netB.Weights }
  let normaliseNablaMatrix =
    List.map (List.map ((*) (nu / miniBatchSize)))
  let nablaNetwork =
    miniBatch
    |> List.map (backpropagate network)
    |> List.fold addNetworks emptyNetwork
    |> fun net -> { Biases = net.Biases |> normaliseNablaMatrix;
                    Weights = net.Weights |> List.map normaliseNablaMatrix }
  { Biases = addMatrices network.Biases nablaNetwork.Biases;
    Weights = List.map2 addMatrices network.Weights nablaNetwork.Weights }

let countCorrectOutputs network data =
  data
  |> List.filter (fun (x, y) ->
        feedForward network x
        |> List.mapi (fun i v -> i, v)
        |> List.maxBy snd
        |> fst
        |> (=) y)
  |> List.length

let displayEpochResults network testData epoch =
  match testData with
  | Some d  -> printfn
                 "Epoch %i: %A / %i"
                 epoch
                 (countCorrectOutputs network d)
                 (List.length d)
  | None    -> printfn "Epoch %i complete" epoch

let stochasticGradientDescent startNetwork trainingData epochs miniBatchSize nu testData rnd =
  let getMiniBatches () =
    trainingData
    |> shuffle rnd
    |> batchesOf miniBatchSize
    |> Seq.map List.ofSeq
  let rec trainNetwork network epoch =
    if epoch = epochs then
      network
    else
      let refinedNetwork =
        getMiniBatches ()
        |> Seq.fold (trainNetworkOnMiniBatch nu) network
      displayEpochResults refinedNetwork testData epoch
      trainNetwork refinedNetwork <| epoch + 1
  trainNetwork startNetwork 0
