module MnistReader

open System
open System.IO

let dataLocation =
  "Users/emmanash/code/fsharp/neural-nets-handwritten-digits/data/"

let openStream fileName =
  new FileStream(
    (sprintf "%s/%s" dataLocation fileName),
    FileMode.Open)

let openReader fileName = new BinaryReader (openStream fileName)

let getDigitImages imageType numberImages convertDigit =
  use imagesReader = openReader <| imageType + "-images-idx3-ubyte"
  use labelsReader = openReader <| imageType + "-labels-idx1-ubyte"

  imagesReader.ReadInt32 () |> ignore
  imagesReader.ReadInt32 () |> ignore
  imagesReader.ReadInt32 () |> ignore
  imagesReader.ReadInt32 () |> ignore

  labelsReader.ReadInt32 () |> ignore
  labelsReader.ReadInt32 () |> ignore

  let numberRows = 28
  let numberCols = 28
  let numberPixels = numberRows * numberCols

  let readImageBytes _ =
    let imageBytes = List.init numberPixels (fun _ -> imagesReader.ReadByte ())
    imageBytes, labelsReader.ReadByte ()

  let parsePixel : byte -> _ =
    Convert.ToInt32
    >> fun intensity -> (double intensity) / 255.0

  let parseImage (imageBytes : byte list, digitByte : byte) =
    let image =
      imageBytes
      |> List.map parsePixel
    let digit =
      digitByte
      |> Convert.ToInt32
      |> convertDigit
    image, digit

  Array.init numberImages readImageBytes
  |> Array.Parallel.map parseImage

let vectorise i : double list =
  List.init 10 (fun k -> if k = i then 1.0 else 0.0)

let trainingSetCount = 5000 // 60000
let testSetCount = 500 // 10000
let getTrainingSet () = getDigitImages "train" trainingSetCount vectorise
let getTestSet () = getDigitImages "t10k" testSetCount id
