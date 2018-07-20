import Base.+, Base.-

mutable struct RewardFunction
    values::Array{<:Number}
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
    RewardFunction(rand(Normal(0,1), features))
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
