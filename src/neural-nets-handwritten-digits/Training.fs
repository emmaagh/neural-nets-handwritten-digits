module Training

open System
open System.Diagnostics
open Util
open Matrix
open Network
open Backpropagation

let trainNetworkOnMiniBatch (nu:float) network miniBatch =
  let nablaNetwork = backpropagate network miniBatch
  let scaledBiases = nablaNetwork.Biases |> List.map ((*) nu)
  let scaledWeights = nablaNetwork.Weights |> List.map ((*) nu)
  { Biases = List.map2 (-) network.Biases scaledBiases;
    Weights = List.map2 (-) network.Weights scaledWeights }

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
    |> Seq.map (fun batch ->
        let batchInput = batch |> Seq.map fst |> fromColumnVectors
        let batchOutput = batch |> Seq.map snd |> fromColumnVectors
        batchInput, batchOutput)
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
