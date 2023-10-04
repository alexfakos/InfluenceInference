__precompile__(true)
#Alex Fakos 2023; alexfakos@icloud.com; Julia 1.6+
module InfluenceInference

import Random

function main(;N::Integer=2000,vrb::Bool=true, t::Number=0.000001, Nboot::Integer=200, rseed::Integer=1266)
    Random.seed!(rseed)
    data = rand(N)
    test_mean(data,t,Nboot,vrb)
    test_sd(data,t,Nboot,vrb)
end


function test_mean(data::AbstractVector, t::Number, Nboot::Integer, vrb::Bool)
    N = length(data)
    Pn   = fill(1/N,N)
    vrb && println("===============================\n ****** Testing inference on the mean. ******** ")
    vrb && println(" Analytical influence function.")
    inference( Inflmean( data )[1]; vrb)
    vrb && println(" Finite difference approximation of influence function with spacing espilon = $(t)")
    inference( gateaux( Tmean, data, Pn; t)[1]; vrb)
    tH = 10 * t
    vrb && println(" Finite difference approximation of influence function with spacing espilon = $(tH)")
    inference( gateaux( Tmean, data, Pn; t=tH)[1]; vrb)
    bse = bootstrap( Tmean, data, N,Nboot)[1]
    vrb && println(" Bootstrap.\n  The standard error is: ", bse, ", and the sample size is N = ", N)
end

function test_sd(data::AbstractVector, t::Number, Nboot::Integer, vrb::Bool)
    N = length(data)
    Pn   = fill(1/N,N)
    vrb && println("===============================\n ****** Testing inference on the standard deviation. ******** ")
    vrb && println(" Analytical influence function.")
    inference( Inflsd( data )[1]; vrb)
    vrb && println(" Finite difference approximation of influence function with spacing epsilon = $(t)")
    inference( gateaux( Tsd, data, Pn; t)[1]; vrb)
    tH = 10 * t
    vrb && println(" Finite difference approximation of influence function with spacing espilon = $(tH)")
    inference( gateaux( Tsd, data, Pn; t=tH)[1]; vrb)
    bse = bootstrap( Tsd, data, N,Nboot)[1]
    vrb && println(" Bootstrap.\n  The standard error is: ", bse, ", and the sample size is N = ", N)
end

"""
`inference(influencevector; vrb::Bool=true)`

Example
=======
gateaux( (x,p) -> sum(x .*p), rand(10000), 10000)[1] |> inference
"""
function inference(infl::AbstractVector; vrb::Bool=true)
    N = length(infl)
    estimatedvariance = (infl'*infl)./N^2
    standarderror = sqrt(estimatedvariance)
    vrb && println("  The standard error is: ", standarderror, ", and the sample size is N = ", N)
    return standarderror
end


#******************************************************************************
# Statistics: Mean and Standard deviation.
#******************************************************************************

function Tmean(X::AbstractVector)
    N = length(X)
    T = (1/N) * sum(X)
    return T
end #Tmean dispatch 1: only X as input
function Tmean(X::AbstractVector, P::AbstractVector)
    return sum( X .* P )
end #Tmean dispatch 2:  X and P as inputs

function Tsd(X::AbstractVector)
    N = length(X)
    m = Tmean(X)
    T = sqrt(  (1/N) * sum( (X .- m) .^2 )  )
    return T
end #Tsd dispatch 1: only X as input
function Tsd(X::AbstractVector, P::AbstractVector)
    average = Tmean(X,P)
    return sqrt(  sum( (X .- average).^2 .* P )  )
end #Tsd dispatch 2:  X and P as inputs


#******************************************************************************
#Numerical finite-difference approximateion of the influence function 
#******************************************************************************

"""
`gateaux(Statistic::Function, DataVector, ProbabilityMassVector ; t::Number=0.000001)`

Returns influencevector, statisticUnderP, N
"""
function gateaux(Tstat::Function, X, P::AbstractVector ; t::Number=0.000001)
    N = length(P)
    infl = fill(NaN,N)
    Tbar = Tstat( X, P)
    perturbedPi = fill(NaN,N)
    for i=1:N
	tperturbPi!(perturbedPi,P,t,N,i)
        infl[i] = (  Tstat( X, perturbedPi ) - Tbar  ) / t
    end
    return infl, Tbar, N
end #gateaux dispatch 1
function tperturbPi!(out::AbstractVector,P::AbstractVector,t::Float64, N::Integer, i::Integer)
    for j=1:N
	i != j ? (out[j] = (1.0 - t) * P[j]) : (out[j] = (1.0 - t) * P[j] + t ) 
    end
end #perturbPi!
"""
`gateaux(Statistic::Function, DataVector, N::Integer ; t::Number=0.000001)`

Sets the probability mass vector to fill(1/N,N).
Returns influencevector, statisticUnderP, N

Examples 
=========
`julia> gateaux( (x,p) -> sum(x .*p), rand(100), 100)`

"""
function gateaux(Tstat::Function, X, N::Integer ; t::Number=0.000001)
    P = fill( 1/N, N)
    return gateaux(Tstat, X, P; t)
end #gateaux dispatch 2


#******************************************************************************
#Bootstrap approximation of the standard error 
#******************************************************************************

function bootstrap(T::Function, data::AbstractVector, Ndata, Nboot::Integer=100)
    #Random.seed!(809)
    S = fill( NaN, Nboot )
    reshuffle = fill(-1, Ndata)
    for j=1:Nboot
        Random.rand!(reshuffle, 1:Ndata)
        S[j] =  T( data[reshuffle] )
    end
    se = Tsd(S)
    return se,S
end


#******************************************************************************
#Analytical Influence Functions
#******************************************************************************

function Inflmean(X::AbstractVector)
    av = Tmean(X)
    N = length(X)
    infl = X .- av
    return infl,av,N
end #Inflmean
function Inflsd(X::AbstractVector)
    s = Tsd(X)
    N = length(X)
    infl = ( X .- Tmean(X) ).^2  .- s^2
    infl = ( 0.5/s ) .* infl
    return infl,s, N
end #Inflsd

end #module
