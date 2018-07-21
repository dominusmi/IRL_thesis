module DPMBIRL

export DPMBIRL, generate_gridworld, generate_trajectories

using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox

immutable Globals
    n_states::Int64
    n_actions::Int64
    n_features::Int64
    n_trajectories::Int64
    actions_i::Array{Int64}
    β::Float64
    γ::Float64
    Pₐ::Array{Array{Float64,2},1}
    χ::Array{MDPHistory}
    ϕ::Array{Float64,2}
end

include("../reward.jl")
include("../cluster.jl")
include("../utilities/gridworld.jl")
include("../utilities/policy.jl")
include("../utilities/general.jl")
include("../utilities/trajectory.jl")



"""
    (proportional) Likelihood function for a single state action
    Normally should have normalisiation, but not important when calculating ∇𝓛
"""
state_action_lh(πᵦ, s,a) = πᵦ[s,a]

"""
    Executes a maximum likelihood inverse reinforcement learning
    mdp:            the problem
    ϕ:              operator to features space
    trajectories:   experts' trajectories
    iterations:     number of iterations to run for
    α:              learning rate
    β:              confidence parameter
    κ:              concentration parameter for DPM
"""
function DPM_BIRL(mdp, ϕ, χ, iterations; α=0.1, κ=1., β=0.5, ground_policy = nothing, verbose=true, update=:ML)

    verbose ? println("Using $(update) update") : nothing

    τ = sqrt(2*α)

    γ = mdp.discount_factor
    states = ordered_states(mdp)

    n_states  = size(states,1)-1
    n_actions = size( actions(mdp),1 )
    n_features = size(ϕ,2)
    n_trajectories = size(χ,1)

    EVD = []

    if verbose && ground_policy !== nothing
        v = policy_evaluation(mdp, ground_policy)
    end

    # Precpmputes transition matrix for all actions
    # (independent of features)
    Pₐ = a2transition.(mdp,actions(mdp))
    actions_i = action_index.(mdp, actions(mdp))

    const glb = Globals(n_states, n_actions, n_features, n_trajectories, actions_i, β, γ, Pₐ, χ, ϕ)

    #### Initialisation ####
    # Initialise clusters
    # K = n_trajectories
    K = 1
    assignements    = collect(1:n_trajectories)
    # assignements    = rand(1:K, n_trajectories)
    assignements = fill(1,n_trajectories)
    N               = map(x->sum(assignements .== x), 1:K)


    # Prepare reward functions
    θs = [sample(RewardFunction, n_features) for i in 1:K]
    for (k,θ) in enumerate(θs)
    #     # Solve mdp with current reward
    #     θ.π  = solve_mdp(mdp, θ)
    #     # Find Boltzmann policy
    #     θ.πᵦ = calπᵦ(mdp, θ.π.qmat, β)
    #
    #     # Prepare variables for gradient
    #     θ.invT = calInvTransition(mdp, θ.πᵦ, γ)
    #     # Calculates value and gradient of trajectory likelihood
    #     θ.𝓛, θ.∇𝓛 = cal∇𝓛(mdp, ϕ, θ.invT, Pₐ, θ.πᵦ, β, χ, n_states, n_actions, n_features, actions_i)
        assigned2cluster = (assignements .== k)
        χₖ = χ[assigned2cluster]
        update_reward!(θ, mdp, χₖ, glb)
    end

    𝓛_traj          = ones(n_trajectories)*1e-5
    c               = Clusters(K, assignements, N, 𝓛_traj, θs)

    update_clusters!(c, mdp, κ, glb)

    log = Dict(:assignements => [], :EVDs => [], :likelihoods => [], :rewards => [])

    for t in 1:iterations
        tic()

        update_clusters!(c, mdp, κ, glb)

        for (k, θ) in enumerate(c.rewards)
            # Find potential new reward
            if update == :langevin_rand
                θ⁻ = θ + α * ∇𝓛 + α * rand(Normal(0,1), n_features)
            else
                θ⁻ = θ + α * θ.∇𝓛
            end

            # Solve everything for potential new reward
            π⁻  = solve_mdp(mdp, θ⁻)
            πᵦ⁻ = calπᵦ(mdp, π⁻.qmat, glb)
            invT⁻ = calInvTransition(mdp, πᵦ⁻, γ)

            # Calculate likelihood and gradient
            assigned2cluster = (c.assignements .== k)
            χₖ = χ[assigned2cluster]
            𝓛⁻, ∇𝓛⁻ = cal∇𝓛(mdp, invT⁻, πᵦ⁻,  χₖ, glb)

            # Do the update
            if update == :ML
                # We simply follow the gradient
                θ.values, θ.𝓛, θ.∇𝓛, θ.invT, θ.π, θ.πᵦ = θ⁻.values, 𝓛⁻, ∇𝓛⁻, invT⁻, π⁻, πᵦ⁻
            elseif update == :langevin || update == :langevin_rand
                # Use result from Choi
                𝓛 += sum(pdf.(Normal(0,1), θ.values))
                𝓛⁻ += sum(pdf.(Normal(0,1), θ⁻.values))
                p =  𝓛⁻ / θ.𝓛 * proposal_distribution(θ⁻, θ, ∇𝓛⁻, τ) / proposal_distribution(θ, θ⁻, ∇𝓛, τ)
                @show p
                if rand() > p
                    θ.values, θ.𝓛, θ.∇𝓛, θ.invT, θ.π, θ.πᵦ = θ⁻.values, 𝓛⁻, ∇𝓛⁻, invT⁻, π⁻, πᵦ⁻
                end
            end
        end

        elapsed = toq()

        # Log EVD
        if verbose
            println("Iteration took $elapsed seconds")
            push!(log[:assignements], copy(c.N))
            push!(log[:likelihoods], map(x->x.𝓛, c.rewards))
            push!(log[:rewards], c.rewards)

            if ground_policy !== nothing
                EVDs = []
                for θ in c.rewards
                    vᵣ = policy_evaluation(mdp, θ.π)
                    push!(EVDs, norm(v-vᵣ))
                end
                push!(log[:EVDs], EVDs)
                vᵣ = policy_evaluation(mdp, θs[1].π)
                push!(EVD, norm(v-vᵣ))
            end
        end
    end

    # Log EVD
    if verbose && ground_policy !== nothing
        # Need to change this to account for features
        πᵣ = solve_mdp(mdp, c.rewards[1])
        vᵣ = policy_evaluation(mdp, πᵣ)
        push!(EVD, norm(v-vᵣ))
        println("Final EVD: $(EVD[end])")
    end

    c, EVD, log
end



# End module
end
