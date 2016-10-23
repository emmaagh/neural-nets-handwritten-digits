module Network

open System
open Matrix

type Network = {
  Biases : Vector list;
  Weights : Matrix list
}

let createNetwork (rnd : Random) layerSizes =
  let getRandomDouble _ = 2.0 * rnd.NextDouble() - 1.0 // Generate random number between -1 and 1
  let getListOfRandomDoubles length = List.init length getRandomDouble
  let biases =
    layerSizes
    |> List.tail
    |> List.map getListOfRandomDoubles
  let weightsBetweenTwoLayers (previousLayerSize, nextLayerSize) =
    List.init nextLayerSize (fun _ -> getListOfRandomDoubles previousLayerSize)
  let weights =
    List.tail layerSizes
    |> Seq.zip layerSizes
    |> List.ofSeq
    |> List.map weightsBetweenTwoLayers
  { Biases = biases;
    Weights = weights }

let sigmoid z =
  1.0 / (1.0 + Math.Pow(Math.E, -z))

let sigmoid' z =
  Math.Pow(Math.E, z) / Math.Pow(Math.Pow(Math.E, z) + 1.0, 2.0)

let cost' =
  List.map2 (fun activation expected -> activation - expected)

let feedForward network input =
  let feedActivationsThroughLayer a weights biases =
    multiply weights a
    |> add biases
    |> List.map sigmoid
  List.fold2 feedActivationsThroughLayer input network.Weights network.Biases

// Returns a network of of partial derivatives of the Cost function
// wrt the biases and the weights
let backpropagate network (input, expectedOutput) =
  let getLayerWeightedInputsAndActivations (_, prevActivations) (weights, biases) =
    let layerWeightedInputs =
      prevActivations
      |> multiply weights
      |> add biases
    let layerActivations = layerWeightedInputs |> List.map sigmoid
    layerWeightedInputs, layerActivations

  // Produce the following list:
  // [
  //  ([], input);
  //  (w. input to layer 2, activations for layer 2);
  //  ..
  //  (w. input to layer n-1, activations for layer n-1);
  //  (w. input to layer n, network output)
  // ]
  let weightedInputsAndActivations =
    network.Biases
    |> List.zip network.Weights
    |> List.scan getLayerWeightedInputsAndActivations ([], input)
  let weightedInputs, activations =
    weightedInputsAndActivations
    |> List.unzip
    |> fun (zs, acts) -> (zs |> Seq.skip 1 |> List.ofSeq), acts

  // (BP1)
  let deltaLast =
    weightedInputs
    |> Seq.last
    |> List.map sigmoid'
    |> List.map2 (*) (cost' (Seq.last activations) expectedOutput)

  // (BP2)
  let backpropagate (previousWeights, weightedInputs) previousDeltas =
     previousDeltas
     |> multiply (transpose previousWeights)
     |> List.map2 (*) (List.map sigmoid' weightedInputs)

  let deltas =
    List.scanBack
      backpropagate
      (Seq.zip (Seq.skip 1 network.Weights) weightedInputs |> List.ofSeq)
      deltaLast

  // (BP4)
  let calculateWeightPartialDerivates layerActivations =
    List.map (fun delta -> List.map ((*) delta) layerActivations)
  let weights =
    Seq.map2 calculateWeightPartialDerivates activations deltas
    |> List.ofSeq

  { Biases = deltas; // (BP3)
    Weights = weights }
