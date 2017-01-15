module Network

open System
open Matrix

type Network = {
  Biases : Vector<float> list
  Weights : Matrix<float> list
}

let createNetwork (rnd : Random) layerSizes =
  let getRandomFloat _ = 2.0 * rnd.NextDouble() - 1.0 // Generate random number between -1 and 1
  let getListOfRandomDoubles length = Array.init length getRandomFloat
  let biases =
    layerSizes
    |> List.tail
    |> List.map (getListOfRandomDoubles >> Vector)
  let weightsBetweenTwoLayers (previousLayerSize, nextLayerSize) =
    Array.init nextLayerSize (fun _ -> getListOfRandomDoubles previousLayerSize)
    |> arrayArrayToMatrix
  let weights =
    List.tail layerSizes
    |> Seq.zip layerSizes
    |> List.ofSeq
    |> List.map weightsBetweenTwoLayers
  { Biases = biases;
    Weights = weights }
