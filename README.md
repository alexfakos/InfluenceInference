# InfluenceInference
Statistical inference using analytical and numerical influence functions.

If you are using this software please cite:

*Fakos, Alexandros (2023). Inference for Misallocation Statistics Using Influence Functions. SSRN paper No. 4501891.*

**Example code** 

To load the julia module/package:

```julia
julia> include("InfluenceInference.jl")
```

To conduct inference on the mean using the numerical approximation to the influence function:

```julia
julia> InfluenceInference.gateaux( (x,p) -> sum(x .* p), rand(10000), 10000)[1] |> InfluenceInference.inference
```

To conduct inference on the standard deviation using the numerical approximation to the influence function:

```julia
InfluenceInference.gateaux( (x,p)->  sqrt(  sum( (x .- sum(x .* p) ).^2 .* p )  ), rand(10000), 10000)[1] |> InfluenceInference.inference
```
