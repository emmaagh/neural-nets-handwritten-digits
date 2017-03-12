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

let getLayerWeightedInputsAndActivations (_, (prevActivations:Matrix<float>)) ((weights:Matrix<float>), biases) =
  let layerWeightedInputs =
    (weights * prevActivations) + biases
  let layerActivations = layerWeightedInputs |> mapMatrix sigmoid
  layerWeightedInputs, layerActivations

// (BP2)
let backpropagateLayer (previousWeights:Matrix<float>, weightedInputs:Matrix<float>) (previousDeltas:Matrix<float>) =
  previousWeights.GetTranspose() * previousDeltas
  |> (.*) (mapMatrix sigmoid' weightedInputs)

// (BP4)
let calculateWeightPartialDerivates (layerActivations:Matrix<float>) (layerDeltas:Matrix<float>) =
  toColumnVectors layerDeltas
  |> List.zip (toColumnVectors layerActivations)
  |> List.map (@*)

let averageVectorOfMatrix matrix =
  let totalVector =
    matrix
    |> toColumnVectors
    |> List.sum
  totalVector / float matrix.Cols

let averageMatrixOfMatrices (matrices:Matrix<float> List) =
  let totalMatrix = matrices |> List.sum
  let length = matrices |> List.length |> float
  totalMatrix / length

// Returns a network of of partial derivatives of the Cost function
// wrt the biases and the weights, using a fully matrix-based approach
let inline backpropagate network (input:Matrix<float>, expectedOutput) =
  let biasesMatrix = matrixOfRepeatedVector (input.Cols)
  let weightedInputsAndActivations =
    network.Biases
    |> List.map biasesMatrix
    |> List.zip network.Weights
    |> List.scan getLayerWeightedInputsAndActivations (Matrix.Zero, input)
  let weightedInputs, activations =
    weightedInputsAndActivations
    |> List.unzip
    |> fun (zs, acts) -> (zs |> Seq.skip 1 |> List.ofSeq), acts

  // (BP1)
  let deltaLast =
    weightedInputs
    |> Seq.last
    |> mapMatrix sigmoid'
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

  let biases = deltas |> List.map averageVectorOfMatrix

  // (BP4)
  let weights =
    Seq.map2 calculateWeightPartialDerivates activations deltas
    |> List.ofSeq
    |> List.map averageMatrixOfMatrices

  { Biases = biases // (BP3)
    Weights = weights }
