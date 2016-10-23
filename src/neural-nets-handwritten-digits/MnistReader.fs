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

  let readInt (reader:BinaryReader) =
    reader.ReadByte ()
    |> Convert.ToInt32

  let getNextPixel _ : double =
    imagesReader
    |> readInt
    |> fun intensity -> (double intensity) / 255.0

  let getNextDigitImage () =
    let image = List.init numberPixels getNextPixel
    let digit = labelsReader |> readInt |> convertDigit
    image, digit

  [|1..numberImages|]
  |> Array.map (fun _ -> getNextDigitImage ())

let vectorise i : double list =
  List.init 10 (fun k -> if k = i then 1.0 else 0.0)

let getTrainingSet () = getDigitImages "train" 60000 vectorise
let getTestSet () = getDigitImages "t10k" 10000 id
