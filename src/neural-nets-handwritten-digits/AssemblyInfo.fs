namespace System
open System.Reflection

[<assembly: AssemblyTitleAttribute("neural-nets-handwritten-digits")>]
[<assembly: AssemblyProductAttribute("neural-nets-handwritten-digits")>]
[<assembly: AssemblyDescriptionAttribute("A neural network which learns how to read handwritten digits")>]
[<assembly: AssemblyVersionAttribute("1.0")>]
[<assembly: AssemblyFileVersionAttribute("1.0")>]
do ()

module internal AssemblyVersionInformation =
    let [<Literal>] Version = "1.0"
    let [<Literal>] InformationalVersion = "1.0"
