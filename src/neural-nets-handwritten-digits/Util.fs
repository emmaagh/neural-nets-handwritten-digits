module Util

open System

let shuffle (rnd : Random) list =
  let length = Array.length list
  let swap (l : _[]) x y =
    let tmp = l.[x]
    l.[x] <- l.[y]
    l.[y] <- tmp
  list
  |> Array.iteri (fun i _ -> swap list i (rnd.Next(i, length)))
  list

let batchesOf size = // This will include a smaller batch at the end - should this be removed? Might also want to consider optimising this code
  Seq.mapi (fun i v -> i / size, v)
  >> Seq.groupBy fst
  >> Seq.map (snd >> Seq.map snd) // Dear god
