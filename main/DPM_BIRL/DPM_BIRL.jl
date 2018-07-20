include("../reward.jl")
include("../utilities/gridworld.jl")
include("../utilities/policy.jl")
include("../utilities/general.jl")


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
"""
function DPM_BIRL(mdp, ϕ, trajectories, iterations; α=0.1, β=0.5, ground_policy = nothing, verbose=true, update=:gradient)

    verbose ? println("Using $(update) update") : nothing

    τ = sqrt(2*α)

    γ = mdp.discount_factor
    states = ordered_states(mdp)

    n_states  = size(states,1)-1
    n_actions = size( actions(mdp),1 )
    n_features = size(ϕ,2)

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

    #### First run ####
    # Solve mdp with current reward
    π  = solve_mdp(mdp, θ)
    # Find Boltzmann policy
    πᵦ = calπᵦ(mdp, π.qmat, β)

    # Prepare variables for gradient
    invT = calInvTransition(mdp, πᵦ, γ)
    ∇𝓛 = zeros(n_features)
    𝓛 = 0.
    # Calculates gradient of trajectory likelihood
    𝓛 = cal∇𝓛!(∇𝓛, mdp, ϕ, invT, Pₐ, πᵦ, β, n_states, n_actions, n_features, actions_i)

    for t in 1:iterations
        tic()

        # Find potential new reward
        if update == :langevin_rand
            θ⁻ = θ + α * ∇𝓛 + τ * rand(Normal(0,1), n_features)
        else
            θ⁻ = θ + α * ∇𝓛
        end

        # Solve everything for potential new reward
        π⁻  = solve_mdp(mdp, θ⁻)
        πᵦ⁻ = calπᵦ(mdp, π⁻.qmat, β)

        invT⁻ = calInvTransition(mdp, πᵦ⁻, γ)
        ∇𝓛⁻ = zeros(n_features)
        𝓛⁻ = cal∇𝓛!(∇𝓛⁻, mdp, ϕ, invT⁻, Pₐ, πᵦ⁻, β, n_states, n_actions, n_features, actions_i)


        # TODO: Do MH update step
        if update == :gradient
            # We simply follow the gradient
            θ = θ⁻
            𝓛, ∇𝓛, invT, π, πᵦ = 𝓛⁻, ∇𝓛⁻, invT⁻, π⁻, πᵦ⁻
        elseif update == :langevin || update == :langevin_rand
            # Use result from Choi
            p =  𝓛⁻ / 𝓛 * proposal_distribution(θ⁻, θ, ∇𝓛⁻, τ) / proposal_distribution(θ, θ⁻, ∇𝓛, τ)
            @show p
            if rand() > p
                θ = θ⁻
                𝓛, ∇𝓛, invT, π, πᵦ = 𝓛⁻, ∇𝓛⁻, invT⁻, π⁻, πᵦ⁻
            end
        end
        # θ = θ⁻

        elapsed = toq()

        # Log EVD
        if verbose
            println("Iteration took $elapsed seconds")
            if ground_policy !== nothing
                # Need to change this to account for features
                vᵣ = policy_evaluation(mdp, π)
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

function cal∇𝓛!(∇𝓛, mdp, ϕ, invT, Pₐ, πᵦ, β, n_states, n_actions, n_features, actions_i)
    𝓛  = 0.
    for k in 1:n_features
        dQₖ = zeros( n_states, n_actions )
        caldQₖ!(dQₖ, mdp, ϕ, invT, Pₐ, πᵦ, k)
        # Calculates total gradient over trajectories


        for (m,trajectory) in enumerate(trajectories)
            for (h,state) in enumerate(trajectory.state_hist[1:end-1])
                sₕ = state_index(mdp, state)
                aₕ = action_index(mdp, trajectory.action_hist[h])

                𝓛 += state_action_lh(πᵦ,sₕ,aₕ)

                dl_dθₖ = β * ( dQₖ[sₕ,aₕ] - sum( [ state_action_lh(πᵦ,sₕ,ai⁻) * dQₖ[sₕ,ai⁻] for ai⁻ in actions_i ] ) )
                ∇𝓛[k] += dl_dθₖ
            end
        end
    end
    𝓛
end
