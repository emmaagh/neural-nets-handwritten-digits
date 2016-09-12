module Matrix

open System

type Vector = double list

type Matrix = double list list

let multiply (m : Matrix) (v : Vector) : Vector =
  m
  |> List.map (List.map2 (*) v >> List.sum)

let add (v1 : Vector) (v2 : Vector) : Vector =
  List.map2 (+) v1 v2

let addMatrices (m1 : Matrix) (m2 : Matrix) : Matrix =
  List.map2 add m1 m2

let rec transpose (m : Matrix) : Matrix =
  match m with
  | row :: rows ->
    match row with
    | col :: cols ->
      let first = List.map List.head m
      let rest = List.map List.tail m |> transpose
      first :: rest
    | [] -> []
  | _ -> []
