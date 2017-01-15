module Backpropagation

open System
open Matrix
open Network

let sigmoid z =
  1.0 / (1.0 + Math.Pow(Math.E, -z))

let sigmoid' z =
  Math.Pow(Math.E, z) / Math.Pow(Math.Pow(Math.E, z) + 1.0, 2.0)

let cost' activations expectedValues =
  activations - expectedValues

let feedForward network input =
  let feedActivationsThroughLayer (a:Vector<float>) (weights:Matrix<float>) (biases:Vector<float>) =
    biases + (weights * a)
    |> map sigmoid
  network.Biases
  |> List.fold2 feedActivationsThroughLayer input network.Weights

let getLayerWeightedInputsAndActivations (_, (prevActivations:Vector<float>)) ((weights:Matrix<float>), biases) =
  let layerWeightedInputs =
    (weights * prevActivations) + biases
  let layerActivations = layerWeightedInputs |> map sigmoid
  layerWeightedInputs, layerActivations

// (BP2)
let backpropagateLayer (previousWeights:Matrix<float>, weightedInputs:Vector<float>) (previousDeltas:Vector<float>) =
  previousWeights.GetTranspose() * previousDeltas
  |> (.*) (map sigmoid' weightedInputs)

// (BP4)
let calculateWeightPartialDerivates (layerActivations:Vector<float>) (layerDeltas:Vector<float>) =
  layerDeltas.ToArray()
  |> Array.map (fun delta -> Array.map ((*) delta) (layerActivations.ToArray()))
  |> arrayArrayToMatrix

// Returns a network of of partial derivatives of the Cost function
// wrt the biases and the weights
let inline backpropagate network (input, expectedOutput) =
  let weightedInputsAndActivations =
    network.Biases
    |> List.zip network.Weights
    |> List.scan getLayerWeightedInputsAndActivations (Vector [||], input)
  let weightedInputs, activations =
    weightedInputsAndActivations
    |> List.unzip
    |> fun (zs, acts) -> (zs |> Seq.skip 1 |> List.ofSeq), acts

  // (BP1)
  let deltaLast =
    weightedInputs
    |> Seq.last
    |> map sigmoid'
    |> (.*) (cost' (Seq.last activations) expectedOutput)

  // (BP2)
  let deltas =
    let zippedWeightsAndInputs =
      weightedInputs
      |> Seq.zip (Seq.skip 1 network.Weights)
      |> List.ofSeq
    List.scanBack
      backpropagateLayer
      zippedWeightsAndInputs
      deltaLast

  // (BP4)
  let weights =
    Seq.map2 calculateWeightPartialDerivates activations deltas
    |> List.ofSeq

  { Biases = deltas // (BP3)
    Weights = weights }
