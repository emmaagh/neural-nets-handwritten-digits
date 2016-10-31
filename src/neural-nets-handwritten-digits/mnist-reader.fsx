#load "MnistReader.fs"

open System.Diagnostics
open MnistReader

let loadDataSets () =
  let stopwatch = Stopwatch.StartNew ()

  getTestSet () |> ignore

  printfn "%f" stopwatch.Elapsed.TotalSeconds

  getTrainingSet () |> ignore

  stopwatch.Stop ()
  printfn "%f" stopwatch.Elapsed.TotalSeconds

loadDataSets ()
