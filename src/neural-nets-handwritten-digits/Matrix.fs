module Matrix

open System

/// Gets the transpose of the 2d array `m`
let inline transpose (m:_[,]) = Array2D.init (m.GetLength 1) (m.GetLength 0) (fun i j -> m.[j, i])

/// Generic vector type
[<NoEquality; NoComparison>]
type Vector<'S when 'S : (static member Zero : 'S)
                and 'S : (static member One : 'S)
                and 'S : (static member (+) : 'S * 'S -> 'S)
                and 'S : (static member (-) : 'S * 'S -> 'S)
                and 'S : (static member (*) : 'S * 'S -> 'S)> =
    | ZeroVector of 'S
    | Vector of 'S[]
    /// ZeroVector
    static member inline Zero = ZeroVector LanguagePrimitives.GenericZero<'S>
    /// Gets the total number of elements of this vector
    member inline v.Length =
        match v with
        | Vector v -> v.Length
        | ZeroVector _ -> 0
    /// Converts this vector to an array
    member inline v.ToArray() =
        match v with
        | Vector v -> v
        | ZeroVector _ -> [||]
    /// Adds vector `a` to vector `b`
    static member inline (+) (a:Vector<'S>, b:Vector<'S>):Vector<'S> =
        match a, b with
        | Vector a, Vector b -> try Vector (Array.map2 (+) a b) with | _ -> invalidArg "" "Cannot add two vectors of different dimensions."
        | Vector _, ZeroVector _ -> a
        | ZeroVector _, Vector _ -> b
        | ZeroVector _, ZeroVector _ -> Vector.Zero
    /// Subtracts vector `b` from vector `a`
    static member inline (-) (a:Vector<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Vector a, Vector b -> try Vector (Array.map2 (-) a b) with | _ -> invalidArg "" "Cannot subtract two vectors of different dimensions."
        | Vector _, ZeroVector _ -> a
        | ZeroVector _, Vector b -> Vector (Array.map (~-) b)
        | ZeroVector _, ZeroVector _ -> Vector.Zero
    /// Computes the inner product (dot / scalar product) of vector `a` and vector `b`
    static member inline (*) (a:Vector<'T>, b:Vector<'T>):'T =
        match a, b with
        | Vector a, Vector b -> try Array.map2 (*) a b |> Array.sum with | _ -> invalidArg "" "Cannot multiply two vectors of different dimensions."
        | Vector _, ZeroVector _ -> LanguagePrimitives.GenericZero<'T>
        | ZeroVector _, Vector _ -> LanguagePrimitives.GenericZero<'T>
        | ZeroVector _, ZeroVector _ -> LanguagePrimitives.GenericZero<'T>
    /// Multiplies vector `a` and vector `b` element-wise (Hadamard product)
    static member inline (.*) (a:Vector<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Vector a, Vector b -> try Vector (Array.map2 (*) a b) with | _ -> invalidArg "" "Cannot multiply two vectors of different dimensions."
        | Vector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector _ -> Vector.Zero
        | ZeroVector _, ZeroVector _ -> Vector.Zero
    /// Multiplies each element of vector `a` by scalar `b`
    static member inline (*) (a:Vector<'T>, b:'T):Vector<'T> =
        match a with
        | Vector a -> Vector (Array.map ((*) b) a)
        | ZeroVector _ -> Vector.Zero
    /// Multiplies each element of vector `b` by scalar `a`
    static member inline (*) (a:'T, b:Vector<'T>):Vector<'T> =
        match b with
        | Vector b -> Vector (Array.map ((*) a) b)
        | ZeroVector _ -> Vector.Zero
    /// Divides each element of vector `a` by scalar `b`
    static member inline (/) (a:Vector<'T>, b:'T):Vector<'T> =
        match a with
        | Vector a -> Vector (Array.map (fun x -> x / b) a)
        | ZeroVector _ -> Vector.Zero

[<NoEquality; NoComparison>]
type Matrix<'T when 'T : (static member Zero : 'T)
                and 'T : (static member One : 'T)
                and 'T : (static member (+) : 'T * 'T -> 'T)
                and 'T : (static member (-) : 'T * 'T -> 'T)
                and 'T : (static member (*) : 'T * 'T -> 'T)
                and 'T : (static member (/) : 'T * 'T -> 'T)
                and 'T : (static member (~-) : 'T -> 'T)> =
    | ZeroMatrix of 'T
    | Matrix of 'T[,]
    /// ZeroMatrix
    static member inline Zero = ZeroMatrix LanguagePrimitives.GenericZero<'T>
    /// Gets the number of rows of this matrix
    member inline m.Rows =
        match m with
        | Matrix m -> m.GetLength 0
        | ZeroMatrix _ -> 0
    /// Gets the number of columns of this matrix
    member inline m.Cols =
        match m with
        | Matrix m -> m.GetLength 1
        | ZeroMatrix _ -> 0
    /// The entry of this matrix at row `i` and column `j`
    member inline m.Item
       with get (i, j) =
          match m with
          | Matrix m -> m.[i, j]
          | ZeroMatrix z -> z
       and set (i, j) v =
           match m with
           | Matrix m -> m.[i, j] <- v
           | ZeroMatrix _ -> ()
    /// Converts this matrix into a 2d array
    member inline m.ToArray2D() =
        match m with
        | Matrix m -> m
        | ZeroMatrix _ -> Array2D.zeroCreate 0 0
    /// Gets the transpose of this matrix
    member inline m.GetTranspose() =
        match m with
        | Matrix m -> Matrix (transpose m)
        | ZeroMatrix z -> ZeroMatrix z
    /// Adds matrix `a` to matrix `b`
    static member inline (+) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb ->
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "" "Cannot add matrices of different sizes."
            Matrix (Array2D.init a.Rows a.Cols (fun i j -> ma.[i, j] + mb.[i, j]))
        | Matrix _, ZeroMatrix _ -> a
        | ZeroMatrix _, Matrix _ -> b
        | ZeroMatrix z, ZeroMatrix _ -> ZeroMatrix z
    /// Subtracts matrix `b` from matrix `a`
    static member inline (-) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb ->
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "" "Cannot subtract matrices of different sizes."
            Matrix (Array2D.init a.Rows a.Cols (fun i j -> ma.[i, j] - mb.[i, j]))
        | Matrix _, ZeroMatrix _ -> a
        | ZeroMatrix _, Matrix b -> Matrix (Array2D.map (~-) b)
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
    /// Multiplies matrix `a` and matrix `b` (matrix product)
    static member inline (*) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb ->
            if (a.Cols <> b.Rows) then invalidArg "" "Cannot multiply two matrices of incompatible sizes."
            Matrix (Array2D.init a.Rows b.Cols (fun i j -> Array.sumBy (fun k -> ma.[i, k] * mb.[k, j]) [|0..(b.Rows - 1)|] ))
        | Matrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix _ -> Matrix.Zero
    /// Computes the matrix-vector product of matrix `a` and vector `b`
    static member inline (*) (a:Matrix<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Matrix ma, Vector vb ->
            if (a.Cols <> b.Length) then invalidArg "" "Cannot compute the matrix-vector product of a matrix and a vector of incompatible sizes."
            Vector (Array.init a.Rows (fun i -> Array.sumBy (fun j -> ma.[i, j] * vb.[j]) [|0..(b.Length - 1)|] ))
        | Matrix _, ZeroVector _ -> Vector.Zero
        | ZeroMatrix _, Vector _ -> Vector.Zero
          | ZeroMatrix _, ZeroVector _ -> Vector.Zero
    /// Multiplies each element of matrix `a` by scalar `b`
    static member inline (*) (a:Matrix<'T>, b:'T):Matrix<'T> =
        match a with
        | Matrix a -> Matrix (Array2D.map ((*) b) a)
        | ZeroMatrix _ -> Matrix.Zero
    /// Multiplies each element of matrix `b` by scalar `a`
    static member inline (*) (a:'T, b:Matrix<'T>):Matrix<'T> =
        match b with
        | Matrix b -> Matrix (Array2D.map ((*) a) b)
        | ZeroMatrix _ -> Matrix.Zero
    /// Multiplies matrix `a` and matrix `b` element-wise (Hadamard product)
    static member inline (.*) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb ->
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "" "Cannot multiply matrices of different sizes."
            Matrix (Array2D.init a.Rows a.Cols (fun i j -> ma.[i, j] * mb.[i, j]))
        | Matrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix _ -> Matrix.Zero
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
        /// Divides each element of matrix `a` by scalar `b`
    static member inline (/) (a:Matrix<'T>, b:'T):Matrix<'T> =
        match a with
        | Matrix a -> Matrix (Array2D.map (fun x -> x / b) a)
        | ZeroMatrix _ -> Matrix.Zero

let inline arrayArrayToMatrix (m:'T[][]):Matrix<'T> = m |> array2D |> Matrix

/// Converts matrix `m` to a 2d array, e.g. from Matrix<float> to float[,]
let inline toArray2D (m:Matrix<'T>):'T[,] = m.ToArray2D()

let vectorOfZeroes (v:Vector<float>) : Vector<float> =
  v.ToArray ()
  |> Array.map (fun _ -> 0.0)
  |> Vector

let inline matrixOfZeroes (m:Matrix<float>) : Matrix<float> =
  let rows = m.Rows
  let cols = m.Cols
  Matrix (array2D (Array.init rows (fun _ -> Array.init cols (fun _ -> 0.0))))

/// Creates a vector whose elements are the results of applying function `f` to each element of vector `v`
let inline map (f:'T->'U) (v:Vector<'T>):Vector<'U> = v.ToArray() |> Array.map f |> Vector

/// Creates a matrix whose entries are the results of applying function `f` to each entry of matrix `m`
let inline mapMatrix (f:'T->'U) (m:Matrix<'T>):Matrix<'U> = m |> toArray2D |> Array2D.map f |> Matrix

let inline toColumnVectors (m:Matrix<'T>) : Vector<'T> list =
  let rows = m.Rows
  let getColumnVector j =
    let column = Array.init rows (fun i -> m.[i, j])
    Vector column
  List.init m.Cols getColumnVector

let inline fromRowVectors (rows:Vector<'T> seq) : Matrix<'T> =
  rows
  |> Seq.map (fun row -> row.ToArray())
  |> Array.ofSeq
  |> arrayArrayToMatrix

let inline fromColumnVectors columns =
  (fromRowVectors columns).GetTranspose ()

let inline matrixOfRepeatedVector (noOfCols:int) (v:Vector<'T>) : Matrix<'T> =
    Array.init noOfCols (fun _ -> v.ToArray())
    |> arrayArrayToMatrix
    |> fun m -> m.GetTranspose ()

// Computes the outer product of a and b
let inline (@*) (a:Vector<'T>, b:Vector<'T>):Matrix<'T> =
    match a, b with
    | Vector a, Vector b -> Matrix (Array2D.init b.Length a.Length (fun i j -> b.[i] * a.[j]))
    | Vector _, ZeroVector _ -> Matrix.Zero
    | ZeroVector _, Vector _ -> Matrix.Zero
    | ZeroVector _, ZeroVector _ -> Matrix.Zero

let inline printMatrix (m:Matrix<'T>) =
  printfn "M %ix%i" m.Cols m.Rows
