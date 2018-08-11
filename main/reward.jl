import Base.+, Base.-, Base.copy

mutable struct RewardFunction
    weights::Array{<:Number}
    π::Policy
    πᵦ::Array{<:AbstractFloat,2}
    invT::Array{<:AbstractFloat,2}
    𝓛::Float64
    ∇𝓛::Array{<:AbstractFloat,1}
    values::Array{<:Number}
    RewardFunction(weights::Array{<:Number}) = new(weights)
    RewardFunction(weights::Array{<:Number},
        π::Policy,
        πᵦ::Array{<:AbstractFloat,2},
        invT::Array{<:AbstractFloat,2},
        𝓛::Float64,
        ∇𝓛::Array{<:AbstractFloat,1}) = new(weights, π, πᵦ, invT, 𝓛, ∇𝓛)
    RewardFunction(weights::Array{<:Number},
        π::Policy, πᵦ::Array{<:AbstractFloat,2},
        invT::Array{<:AbstractFloat,2}, 𝓛::Float64,
        ∇𝓛::Array{<:AbstractFloat,1}, v::Vector{<:Number}) = new(weights, π, πᵦ, invT, 𝓛, ∇𝓛, v)
end


function copy(r::RewardFunction)
    RewardFunction(copy(r.weights), DiscreteValueIteration.Policy(r.π),
                    copy(r.πᵦ), copy(r.invT), copy(r.𝓛), copy(r.∇𝓛), copy(r.values))
end

function +(r::RewardFunction, values::Array{<:AbstractFloat})
    RewardFunction(r.weights+values)
end

function -(r::RewardFunction, values::Array{<:AbstractFloat})
    RewardFunction(r.weights-values)
end

""" Get weights """
r2weights(r::RewardFunction) = r.weights


"""
    Sample a new reward value for every state from gaussian
"""
function sample(::Type{RewardFunction}, features)
    # Choi states that he sets 80% of reward values to zero
    weights = rand(Normal(0,1), features)
    # values = [ rand()<0.2?value:0.0 for value in values]
    RewardFunction(weights)
end

"""
    Calculates proposal value of new reward
"""
function proposal_distribution(r₁::RewardFunction, r₂::RewardFunction, ∇logTarget::Array, τ)
    # D = size(r₁.values,1)
    g = r₂.weights - r₁.weights - 0.25*τ^2 * ∇logTarget
    g = -inv(-2*τ^2) * norm(g)^2
    # This is the correct calculation, but the initial constant cancels out
    # g = inv( (2*π*τ^2)^(D/2) ) * exp(g)
    g
end


function update_reward!(θ::RewardFunction, mdp, χₖ, glb::Globals)
    global globals
    # Update real space from features
    θ.values = glb.ϕ * θ.weights
    # Solve mdp with current reward
    θ.π  = solve_mdp(mdp, θ)
    # Find Boltzmann policy
    θ.πᵦ = calπᵦ(mdp, θ.π.qmat, glb)
    # Prepare variables for gradient
    θ.invT = calInvTransition(mdp, θ.πᵦ, glb.γ)
    # Calculates value and gradient of trajectory likelihood
    θ.𝓛 = cal𝓛(mdp, θ.πᵦ, χₖ, glb)
    θ.∇𝓛 = cal∇𝓛(mdp, θ.invT, θ.πᵦ, χₖ, glb)
end

"""
    Return normal prior of reward and its gradient
"""
function log_prior(r::RewardFunction)
    # variance = σ²
    var = 1.0
    sum(-(r.weights'*r.weights)./(2*var^2))+log(inv(sqrt(2*3.1415*var))), -r.weights ./ var
end


function values(r::RewardFunction, ϕ)
    ϕ*r.weights
end
