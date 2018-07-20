using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox


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

    # Initialise random reward function
    θ = sample(RewardFunction, n_features)

    EVD = []

    if verbose && ground_policy !== nothing
        v = policy_evaluation(mdp, ground_policy)
    end

    # Precpmputes transition matrix for all actions
    # (independent of features)
    Pₐ = a2transition.(mdp,actions(mdp))
    actions_i = action_index.(mdp, actions(mdp))

    #### Initialisation ####
    # Initialise clusters
    K = 1

    # Prepare reward functions
    # θs              = [sample(RewardFunction, mdp.size_x*mdp.size_y) for i in 1:K]
    # for θ in θs
    #     # Solve mdp with current reward
    #     θ.π  = solve_mdp(mdp, θ)
    #     # Find Boltzmann policy
    #     θ.πᵦ = calπᵦ(mdp, π.qmat, β)
    #
    #     # Prepare variables for gradient
    #     θ.invT = calInvTransition(mdp, πᵦ, γ)
    #     # Calculates value and gradient of trajectory likelihood
    #     θ.𝓛, θ.∇𝓛 = cal∇𝓛(mdp, ϕ, θ.invT, Pₐ, θ.πᵦ, β, χ, n_states, n_actions, n_features, actions_i)
    # end
    #
    # assignements    = rand(1:K, n_trajectories)
    # N               = map(x->sum(assignements .== x), 1:K)
    # 𝓛               = ones(K)*1e-5
    # c               = Clusters(K, assignements, N, 𝓛, θs)
    #
    # update_clusters!(c, χ, κ, β)

    # Solve mdp with current reward
    θ.π  = solve_mdp(mdp, θ)
    # Find Boltzmann policy
    θ.πᵦ = calπᵦ(mdp, θ.π.qmat, β)

    # Prepare variables for gradient
    θ.invT = calInvTransition(mdp, θ.πᵦ, γ)
    # Calculates value and gradient of trajectory likelihood
    θ.𝓛, θ.∇𝓛 = cal∇𝓛(mdp, ϕ, θ.invT, Pₐ, θ.πᵦ, β, χ, n_states, n_actions, n_features, actions_i)

    for t in 1:30
        tic()

        # Find potential new reward
        if update == :langevin_rand
            θ⁻ = θ + α * ∇𝓛 + α * rand(Normal(0,1), n_features)
        else
            θ⁻ = θ + α * θ.∇𝓛
        end

        # Solve everything for potential new reward
        π⁻  = solve_mdp(mdp, θ⁻)
        πᵦ⁻ = calπᵦ(mdp, π⁻.qmat, β)

        invT⁻ = calInvTransition(mdp, πᵦ⁻, γ)
        𝓛⁻, ∇𝓛⁻ = cal∇𝓛(mdp, ϕ, invT⁻, Pₐ, πᵦ⁻, β, χ, n_states, n_actions, n_features, actions_i)

        # Do the update
        if update == :ML
            # We simply follow the gradient
            θ.values, θ.𝓛, θ.∇𝓛, θ.invT, θ.π, θ.πᵦ = θ⁻.values, 𝓛⁻, ∇𝓛⁻, invT⁻, π⁻, πᵦ⁻
        elseif update == :langevin || update == :langevin_rand
            # Use result from Choi
            𝓛 += sum(pdf.(Normal(0,1), θ.values))
            𝓛⁻ += sum(pdf.(Normal(0,1), θ⁻.values))
            p =  𝓛⁻ / 𝓛 * proposal_distribution(θ⁻, θ, ∇𝓛⁻, τ) / proposal_distribution(θ, θ⁻, ∇𝓛, τ)
            @show p
            if rand() > p
                θ.values, θ.𝓛, θ.∇𝓛, θ.invT, θ.π, θ.πᵦ = θ⁻.values, 𝓛⁻, ∇𝓛⁻, invT⁻, π⁻, πᵦ⁻
            end
        end

        elapsed = toq()

        # Log EVD
        if verbose
            println("Iteration took $elapsed seconds")
            if ground_policy !== nothing
                # Need to change this to account for features
                vᵣ = policy_evaluation(mdp, θ.π)
                push!(EVD, norm(v-vᵣ))
            end
        end
    end

    # Log EVD
    if verbose && ground_policy !== nothing
        # Need to change this to account for features
        πᵣ = solve_mdp(mdp, θ)
        vᵣ = policy_evaluation(mdp, πᵣ)
        push!(EVD, norm(v-vᵣ))
        println("Final EVD: $(EVD[end])")
    end

    θ, EVD
end




function cal∇𝓛(mdp, ϕ, invT, Pₐ, πᵦ, β, χ, n_states, n_actions, n_features, actions_i)
    𝓛  = 0.
    ∇𝓛 = zeros(n_features)
    for k in 1:n_features
        dQₖ = zeros( n_states, n_actions )
        caldQₖ!(dQₖ, mdp, ϕ, invT, Pₐ, πᵦ, k)

        # Calculates total gradient over trajectories
        for (m,trajectory) in enumerate(χ)
            for (h,state) in enumerate(trajectory.state_hist[1:end-1])
                sₕ = state_index(mdp, state)
                aₕ = action_index(mdp, trajectory.action_hist[h])

                𝓛 += state_action_lh(πᵦ,sₕ,aₕ)

                dl_dθₖ = β * ( dQₖ[sₕ,aₕ] - sum( [ state_action_lh(πᵦ,sₕ,ai⁻) * dQₖ[sₕ,ai⁻] for ai⁻ in actions_i ] ) )
                ∇𝓛[k] += dl_dθₖ
            end
        end
    end
    𝓛, ∇𝓛
end
