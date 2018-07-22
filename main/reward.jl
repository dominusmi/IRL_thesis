import Base.+, Base.-

mutable struct RewardFunction
    values::Array{<:Number}
    π::Policy
    πᵦ::Array{<:AbstractFloat,2}
    invT::Array{<:AbstractFloat,2}
    𝓛::Float64
    ∇𝓛::Array{<:AbstractFloat,1}
    RewardFunction(values::Array{<:Number}) = new(values)
end

function +(r::RewardFunction, values::Array{<:AbstractFloat})
    RewardFunction(r.values+values)
end

function -(r::RewardFunction, values::Array{<:AbstractFloat})
    RewardFunction(r.values-values)
end

"""
    Sample a new reward value for every state from gaussian
"""
function sample(::Type{RewardFunction}, features)
    # Choi states that he sets 80% of reward values to zero
    values = rand(Normal(0,1), features)
    # values = [ rand()<0.2?value:0.0 for value in values]
    RewardFunction(values)
end

"""
    Calculates proposal value of new reward
"""
function proposal_distribution(r₁::RewardFunction, r₂::RewardFunction, ∇logTarget::Array, τ)
    D = size(r₁.values,1)
    g = r₁.values - r₂.values - 0.5*τ^2 * ∇logTarget
    g = inv(-2*τ^2) * norm(g)^2
    # This is the correct calculation, but the initial constant cancels out
    # g = inv( (2*π*τ^2)^(D/2) ) * exp(g)
    # @show g
    # exp(g)
    g
end


function update_reward!(θ::RewardFunction, mdp, χₖ, glb::Globals)
    global globals
    # Solve mdp with current reward
    θ.π  = solve_mdp(mdp, θ)
    # Find Boltzmann policy
    θ.πᵦ = calπᵦ(mdp, θ.π.qmat, glb)

    # Prepare variables for gradient
    θ.invT = calInvTransition(mdp, θ.πᵦ, glb.γ)
    # Calculates value and gradient of trajectory likelihood
    θ.𝓛, θ.∇𝓛 = cal∇𝓛(mdp, θ.invT, θ.πᵦ, χₖ, glb)
end
