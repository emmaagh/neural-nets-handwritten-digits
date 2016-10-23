#load "MnistReader.fs"

open MnistReader

getTestSet ()
|> Seq.map snd
|> Seq.max
|> printfn "%A"

getTrainingSet ()
|> Seq.length
|> printfn "%i"
