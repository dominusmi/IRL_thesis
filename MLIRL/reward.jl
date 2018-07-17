import Base.+, Base.-

immutable RewardFunction
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
