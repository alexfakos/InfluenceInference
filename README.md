# InfluenceInference
Statistical inference using analytical and numerical influence functions.

If you are using this software please cite:

Fakos, Alexandros (2023). Inference for Misallocation Statistics Using Influence Functions. SSRN paper No. 4501891.

Example.

To conduct inference on the mean using the numerical approximation to the influence function:

```
julia> include("InfluenceInference.jl")
julia> InfluenceInference.gateaux( (x,p) -> sum(x .* p), rand(10000), 10000)[1] |> InfluenceInference.inference
```
