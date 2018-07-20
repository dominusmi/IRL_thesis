import Base.deleteat!

mutable struct Clusters
    K::Int64                        # Cluster index
    assignements::Array{<:Integer}  # cₘ in paper
    N::Array{<:Integer}             # Number of trajectories per cluster
    𝓛::Array{<:AbstractFloat}       # Current likelihood of cluster given reward
    rewards::Array{RewardFunction}  # Reward function
end
function Clusters(assignements)
    K = unique(unique(assignements))
    Clusters(K, assignements)
end

abstract type Likelihood end


"""
   Samples a potential new clustering in the range [1, 2, ..., K+1].

    Samples probabilities from
    Dir(``\\frac{N\\i}{\\alpha-1+\\sum_i N\\i}, \frac{\\alpha}{alpha-1+\\sum_i N_i}``)
    and then multinomial over the resulting probabilities
"""
function sample(c::Clusters, m::Integer, α::AbstractFloat)
    αs = Vector{Float64}(c.K+1)
    ∑N = size(c.assignements,1)
    cₘ = c.assignements[m]
    for k in 1:c.K
        Nₖ = c.N[k]
        if k == cₘ
            Nₖ -= 1
            Nₖ = (Nₖ == 0 ? 1e-5 : Nₖ)
        end
        αs[k] = Nₖ / (α-1+∑N)
    end
    αs[end] = α / (α-1+∑N)
    indmax(rand(Dirichlet(αs),1))
end

"""
    Removes a cluster and redistributes indexing so that it stays a continuous
    discrete interval
    e.g. need to remove 3rd cluster
        1,2,3,4,5
        -> 1,2,3,4
        where 3 became 4, and 4 became 5
"""
function deleteat!(c::Clusters, index::Integer)
    deleteat!(c.N, index)
    deleteat!(c.rewards, index)
    c.K -= 1
    temp = c.assignements .> index
    c.assignements[temp] -= 1
    return
end

"""
    Accepts proposition with probability proportional to the likelihood of the new
    assignement against the likelihood of the old one
"""
function accept_proposition(::Type{Likelihood}, new_l::AbstractFloat, l::AbstractFloat)
    if new_l > l
        return true
    end
    P = new_l / l
    rand() > P
end


"""
    Updates cluster assignement following Choi's paper
    c:      clusters
    χ:      trajectories
    κ:      concentration for DPM
    η:      "confidence" of trajectories (Boltzmann temperature)
"""
function update_clusters!(clusters::Clusters, χ::Array{MDPHistory}, κ, η)
    # Permute trajectory and clusters to avoid any bias
    trajectoryₚ = randperm(size(χ,1))

    # Sample new clusters
    for m in trajectoryₚ

        new_cluster = false
        cₘ   = clusters.assignements[m] # Current assignement
        cₘ⁻  = sample(clusters,m,κ)     # Potential new assignement

        if cₘ⁻ == clusters.K+1
            # If new cluster, sample new reward
            r⁻ = sample(RewardFunction, mdp.size_x * mdp.size_y)
            new_cluster = true
        else
            # Otherwise "load" current reward function
            r⁻ = clusters.rewards[cₘ⁻]
        end

        # Calculate likelihood
        𝓛      = trajectory_likelihood(mdp, χ[m], clusters.rewards[cₘ].π; η=η)
        𝓛⁻     = trajectory_likelihood(mdp, χ[m], r⁻; η=η)
        accept = accept_proposition(Likelihood, 𝓛⁻, 𝓛)

        # Update if accepted
        if accept
            # Update likelihood of trajectory
            clusters.𝓛[m] = 𝓛⁻

            # Update cluster and reward assignements
            if new_cluster
                # Add new cluster
                push!(clusters.rewards, r⁻)
                push!(clusters.assignements, clusters.K+1)
                push!(clusters.N,1)
                clusters.K += 1
                clusters.N[cₘ] -= 1
            else
                # Update clusters to new assignement
                clusters.N[cₘ] -= 1
                clusters.N[cₘ⁻] += 1
                clusters.assignements[m] = cₘ⁻
            end
        end
        # Remove empty clusters
        if (to_remove = findfirst(x->x==0, clusters.N))>0
            remove!(clusters, to_remove)
        end
    end
end
