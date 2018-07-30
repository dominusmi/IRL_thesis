module DPMBIRL

export DPMBIRL, generate_gridworld, generate_trajectories

using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox
using JLD

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

# Logs a EVD matrix, where the rows are the ground-truths,
# and the columns are the EVD with respect to the reward functions
function log_evd!(_log, mdp, θs, ground_truth)
    # EVD is how well does a policy generated by r' behave on an MDP
    # with the true reward function r
    vs = ground_truth[:vs]
    EVD_matrix = zeros(size(vs,1), size(θs,1))
    for (i,v) in enumerate(vs)
        # Make the MDP with the real reward values of the ith agent
        tmp_mdp = copy(mdp)
        tmp_mdp.reward_values = ground_truth[:rewards][i]
        for (j,θ) in enumerate(θs)
            # Solve the MDP with our reward function and get the optimal π
            πᵣ = solve_mdp(mdp, θ)
            # Check how well does π work w.r.t. the optimal value function
            vᵣ = policy_evaluation(tmp_mdp, πᵣ)
            EVD_matrix[i,j] = norm(v - vᵣ,2)
        end
    end
    push!(_log, EVD_matrix)
end


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
    burn_in:        number of iterations not to record (at the beginning)
"""
function DPM_BIRL(mdp, ϕ, χ, iterations; α=0.1, κ=1., β=0.5, ground_truth = nothing, verbose=true, update=:ML, burn_in=5, use_clusters=true, path_to_file=nothing, seed=1)

    srand(seed)
    verbose ? println("Using $(update) update") : nothing

    if path_to_file !== nothing
        path_to_file = "$path_to_file/$(update)_$(seed)_$(use_clusters).jld"
        f = jldopen(path_to_file, "w")
        close(f)
    end

    τ = sqrt(2*α)

    γ = mdp.discount_factor
    states = ordered_states(mdp)

    n_states  = size(states,1)-1
    n_actions = size( actions(mdp),1 )
    n_features = size(ϕ,2)
    n_trajectories = size(χ,1)

    # Precpmputes transition matrix for all actions
    # (independent of features)
    Pₐ = a2transition.(mdp,actions(mdp))
    actions_i = action_index.(mdp, actions(mdp))

    const glb = Globals(n_states, n_actions, n_features, n_trajectories, actions_i, β, γ, Pₐ, χ, ϕ)

    #### Initialisation ####
    # Initialise clusters
    # K = n_trajectories
    # K = 5
    K = 1
    # assignements    = collect(1:n_trajectories)
    assignements    = rand(1:K, n_trajectories)
    # assignements = fill(1,n_trajectories)

    N = map(x->sum(assignements .== x), 1:K)


    # Prepare reward functions
    θs = [sample(RewardFunction, n_features) for i in 1:K]
    for (k,θ) in enumerate(θs)
        assigned2cluster = (assignements .== k)
        χₖ = χ[assigned2cluster]
        update_reward!(θ, mdp, χₖ, glb)
    end

    𝓛_traj = ones(n_trajectories)*1e-5
    c      = Clusters(K, assignements, N, 𝓛_traj, θs)

    use_clusters ? update_clusters!(c, mdp, κ, glb) : nothing

    _log = Dict(:assignements => [], :EVDs => [], :likelihoods => [], :rewards => [], :clusters=>[], :acceptance_probability=>[], :acc_prob=>[])

    σ = eye(n_features)*τ
    burned = 0
    probabilities = []
    for t in 1:iterations
        changed = false
        tic()

        if use_clusters
            updated_clusters_id = Set()
            updated_clusters_id = update_clusters!(c, mdp, κ, glb)
            verbose ? println("Clusters changed: $(length(updated_clusters_id)) of $(c.K)") : nothing
        end

        for (k, θ) in enumerate(c.rewards)
            # Get the clusters' trajectories
            assigned2cluster = (c.assignements .== k)
            χₖ = χ[assigned2cluster]

            # Update likelihood and gradient to current cluster
            # θ.𝓛 = cal𝓛(mdp, θ.πᵦ, χₖ)
            # θ.∇𝓛 = cal∇𝓛(mdp, θ.invT, θ.πᵦ,  χₖ, glb)
            if use_clusters && k ∈ updated_clusters_id
                θ.𝓛 = cal𝓛(mdp, θ.πᵦ, χₖ, glb)
                θ.∇𝓛 = cal∇𝓛(mdp, θ.invT, θ.πᵦ,  χₖ, glb)
            elseif update == :MH
                θ.𝓛 = cal𝓛(mdp, θ.πᵦ, χₖ, glb)
            else
                θ.𝓛 = cal𝓛(mdp, θ.πᵦ, χₖ, glb)
                θ.∇𝓛 = cal∇𝓛(mdp, θ.invT, θ.πᵦ,  χₖ, glb)
            end

            # Find potential new reward
            if update == :langevin_rand
                ϵ = rand(Normal(0,1), n_features)
                indeces = rand(n_features) .< 0.2
                ϵ[indeces] = 0.0
                θ⁻ = θ + α*θ.∇𝓛 + τ*ϵ
                θ⁻.values ./= sum(abs.(θ⁻.values))
            elseif update == :MH
                # ϵ = rand(Normal(0,1), n_features)
                ϵ = rand(MultivariateNormal(σ))
                θ⁻ = θ + ϵ
            else
                θ⁻ = θ + α*θ.∇𝓛
                θ⁻.values ./= sum(abs.(θ⁻.values))
            end

            # Solve everything for potential new reward
            π⁻  = solve_mdp(mdp, θ⁻)
            πᵦ⁻ = calπᵦ(mdp, π⁻.qmat, glb)
            𝓛⁻ = cal𝓛(mdp, πᵦ⁻, χₖ, glb)

            if update !== :MH
                invT⁻ = calInvTransition(mdp, πᵦ⁻, γ)
                ∇𝓛⁻ = cal∇𝓛(mdp, invT⁻, πᵦ⁻,  χₖ, glb)
            end


            # Do the update
            p = 0.
            if update == :ML
                # We simply follow the gradient
                # logPrior⁻, ∇logPrior⁻ = log_prior(θ⁻)
                # 𝓛⁻ += logPrior⁻
                # ∇𝓛⁻ += ∇logPrior⁻
                p=1.0
                println("log 𝓛: $(@sprintf("%.2f", θ.𝓛)), log 𝓛⁻: $(@sprintf("%.2f", 𝓛⁻))")
                θ.values, θ.𝓛, θ.∇𝓛, θ.invT, θ.π, θ.πᵦ = θ⁻.values, 𝓛⁻, ∇𝓛⁻, invT⁻, π⁻, πᵦ⁻
            elseif update == :MH
                logPrior, ~ = log_prior(θ)
                logPrior⁻, ~ = log_prior(θ⁻)
                θ.𝓛 += logPrior
                𝓛⁻ += logPrior⁻
                ∇𝓛⁻ = zeros(0)
                invT⁻ = zeros(0,0)
                println("log 𝓛: $(@sprintf("%.2f", θ.𝓛)), log 𝓛⁻: $(@sprintf("%.2f", 𝓛⁻))")

                p = exp(𝓛⁻ - θ.𝓛)
                # println("   current p: $p")

            elseif update == :langevin || update == :langevin_rand
                # Use result from Choi

                logPrior, ∇logPrior = log_prior(θ)
                logPrior⁻, ∇logPrior⁻ = log_prior(θ⁻)


                println("    ante-prior log 𝓛: ($(@sprintf("%.2f", θ.𝓛)), ∇log𝓛: ($(norm(θ.∇𝓛)), log 𝓛⁻: $(@sprintf("%.2f", 𝓛⁻)), ∇log𝓛: ($(@sprintf("%.2f", norm(∇𝓛⁻)))")

                θ.𝓛 += logPrior
                θ.∇𝓛 += ∇logPrior
                𝓛⁻ += logPrior⁻
                ∇𝓛⁻ += ∇logPrior⁻

                println("    post-prior log 𝓛: ($(@sprintf("%.2f", θ.𝓛)), ∇log𝓛: ($(norm(θ.∇𝓛)), log 𝓛⁻: $(@sprintf("%.2f", 𝓛⁻)), ∇log𝓛: ($(@sprintf("%.2f", norm(∇𝓛⁻)))")


                #### CHOI SHIT ####

                # a = ϵ + τ/2*(θ.∇𝓛 + ∇𝓛⁻)
                # a = exp(-0.5*sum(a.^2))*exp(𝓛⁻);
                # b = exp(-0.5 * sum(ϵ.^2) ) * exp( θ.𝓛 )

                # @show θ.𝓛, 𝓛⁻, a, b, norm(ϵ + τ/2*(θ.∇𝓛 - ∇𝓛⁻))^2

                # p = a/b

                #### CURRENT WORKING VERSION ####
                logpd⁻ = proposal_distribution(θ⁻, θ, ∇𝓛⁻, τ)
                logpd = proposal_distribution(θ, θ⁻, θ.∇𝓛, τ)

                # log_coef = log(inv(2*3.1415*τ^2)^(n_features/2))

                println("log 𝓛: ($(@sprintf("%.2f", θ.𝓛)), log 𝓛⁻: $(@sprintf("%.2f", 𝓛⁻)), logpd: $(@sprintf("%.2f", logpd)), logpd⁻: $(@sprintf("%.2f", logpd⁻)))")
                # print("𝓛: ($(@sprintf("%.2f", exp(θ.𝓛))), 𝓛⁻ $(@sprintf("%.2f", exp(𝓛⁻))), $(@sprintf("%.2f", log_coef+logpd)), $(@sprintf("%.2f", log_coef+logpd⁻)))")


                p = exp(𝓛⁻-θ.𝓛 + logpd⁻-logpd)
                # p =  (𝓛⁻/θ.𝓛) * logpd⁻ / logpd
                # p =  𝓛⁻ / θ.𝓛 * logpd⁻ / logpd


                # p = percentage_likelihood * logpd⁻ / logpd
                # p = exp( 𝓛⁻ + logpd⁻ - θ.𝓛 - logpd)
                println("   current p: $p")
                println("difference old-new: $(norm(θ.values-θ⁻.values))")
                # println("   real p:    $( exp(𝓛⁻ - θ.𝓛) * exp(log_coef+logpd⁻ - log_coef-logpd))")
            end
            if rand() < p
                θ.values, θ.𝓛, θ.∇𝓛, θ.invT, θ.π, θ.πᵦ = θ⁻.values, 𝓛⁻, ∇𝓛⁻, invT⁻, π⁻, πᵦ⁻
                changed = true
                burned += 1
            end
            push!(_log[:acc_prob], p)
        end

        elapsed = toq()

        if changed
            println("Burned: $burned")
        end

        # Log EVD
        verbose ? println("Iteration number $t took $elapsed seconds") : nothing
        if burned > burn_in
            # push!(_log[:assignements], copy(c.N))
            if path_to_file == nothing
                push!(_log[:likelihoods], map(x->x.𝓛, c.rewards))
                push!(_log[:rewards], copy.(c.rewards))
                use_clusters ? push!(_log[:clusters], copy(c)) : nothing

                if ground_truth !== nothing
                    log_evd!(_log[:EVDs], mdp, c.rewards, ground_truth)
                    verbose ? show(_log[:EVDs][end]) : nothing
                end
            elseif path_to_file !== nothing && changed
                f = jldopen(path_to_file, "r+")
                write(f, "reward_$burned", c.rewards[1].values)
                write(f, "likelihood_$burned", c.rewards[1].𝓛)
                close(f)
            end
        elseif burned < burn_in
            push!(_log[:rewards], copy(c.rewards))
        elseif burned == burn_in
            push!(_log[:rewards], copy(c.rewards))
            println("Finished burn in")
            rewards = zeros(burn_in, n_features)
            @show size(_log[:rewards])
            for i in 1:burn_in
                rewards[i,:] = _log[:rewards][i][1].values
            end
            # σ = σ .* [sqrt(cov(rewards[rewards[:,1].!==0.0,i])) for i in 1:n_features]
            _log[:rewards] = []
            # @show σ
            # println("Found new covariance, sample: $(σ[1:3,1:3])")
            burn_in = 0
            burned = 0
        end
    end

    if path_to_file !== nothing
        f = jldopen(path_to_file, "r+")
        write(f, "acceptance_probabilities", _log[:acc_prob])
        close(f)
    end

    c, _log
end



# End module
end
