include("reward.jl")
include("utilities/gridworld.jl")
include("utilities/policy.jl")

function calQᵣ(mdp, r, β = 0.5)
    states = ordered_states(mdp)

    Q = zeros(size(states,1), size(actions(mdp),1))
    policy = solve_mdp(mdp, r)
    γ = mdp.discount_factor

    for s in states
        si = state_index(s)
        for a in actions(mdp)
            ai = action_index(a)
            states⁻, p = transitions
            _sumS = 0.

            for (j,s⁻) in enumerate(states⁻)
                si⁻ = state_index(s⁻)
                _sumQ = 0
                for a⁻ in actions(mdp, s⁻)
                    _softmax = exp( β* policy.qmat[si⁻,action_index(a⁻)]) / sum(β* exp.(policy.qmat[si⁻,:]))
                    _sumQ += policy.qmat[si⁻,action_index(a⁻)] * _softmax
                end
                _sumS += p[j] * _sumQ
            end

            Q[si,ai] = reward(mdp, s, a) + γ * _sumS
        end
    end
    policy, Q
end


function calπᵦ(mdp, Q, β=0.5)
    states = ordered_states(mdp)
    πᵣ = zeros(size(states,1)-1, size(actions(mdp),1))

    for s in states[1:end-1]
        si = state_index(mdp,s)
        softmax_denom = sum(exp.(β*Q[si,:]))
        for a in actions(mdp)
            ai = action_index(mdp,a)
            πᵣ[si,ai] = exp(β*Q[si,ai]) / softmax_denom
        end
    end
    πᵣ
end

function π2transition(mdp, π)
    n_states = size(π,1)
    T = zeros(n_states, n_states)

    for s in 1:n_states
        nbs = state_neighbours(mdp, s)
        for (i,nb) in enumerate(nbs)
            T[s,nb] = π[s,i]
        end
    end
    T
end

"""
    Given an action, generates the SxS probability transition matrix Pₐ
"""
function a2transition(mdp, a)
    states = ordered_states(mdp)
    n_states = size(states,1)-1
    Pₐ = zeros(n_states, n_states)

    for s in states[1:end-1]
        si = state_index(mdp,s)
        states⁻ = transition(mdp, s, a)
        states⁻, p = states⁻.vals, states⁻.probs
        for (j,s⁻) in enumerate(states⁻)
            if isterminal(mdp, s⁻)
                continue
            end
            s⁻ = POMDPModels.inbounds(mdp, s⁻) ? s⁻ : s
            si⁻ = state_index(mdp, s⁻)
            Pₐ[si, si⁻] = p[j]
        end
    end
    Pₐ
end


function calInvTransition(mdp, πᵦ, γ)
    n_states = size(πᵦ,1)
    I = eye(n_states)
    Pₚᵢ = π2transition(mdp, πᵦ)
    inv( I-γ*Pₚᵢ )
end

function MLIRL(mdp, ϕ, trajectories, iterations, learning_rate; ground_policy = nothing)
    srand(1)
    θ = sample(RewardFunction, size(ϕ,2))
    γ = mdp.discount_factor
    states = ordered_states(mdp)

    n_states  = size(states,1)-1
    n_actions = size( actions(mdp),1 )
    n_features = size(ϕ,2)

    EVD = []

    if ground_policy !== nothing
        v = policy_evaluation(mdp, ground_policy)
    end


    β = 0.5
    for t in 1:iterations
        # π, Qᵣ   = calQᵣ(mdp, θ)
        π  = solve_mdp(mdp, θ)
        πᵦ = calπᵦ(mdp, π.qmat, β)

        # Log EVD
        if ground_policy !== nothing
            # Need to change this to account for features
            πᵣ = solve_mdp(mdp, θ)
            vᵣ = policy_evaluation(mdp, πᵣ)
            push!(EVD, norm(v-vᵣ))
        end


        invT = calInvTransition(mdp, πᵦ, γ)
        actions_i = action_index.(mdp, actions(mdp))
        dQ = zeros(n_states, n_actions)

        ∇𝓛 = zeros(n_features)

        for k in 1:n_features

            dQ = zeros( n_states, n_actions )
            πₐ = zeros(size(states,1)-1)

            # π "marginal"
            for s in states[1:end-1]
                si      = state_index(mdp, s)
                πₐ[si]  = sum( πᵦ[si,:] ) * ϕ[si,k]
            end

            # dQ for each action
            for a in actions(mdp)
                Pₐ = a2transition(mdp, a)
                ai = action_index(mdp,a)
                dQ[:,ai] = ϕ[:,k] + γ * Pₐ * invT * πₐ
            end

            for (m,trajectory) in enumerate(trajectories)
                for (h,state) in enumerate(trajectory.state_hist[1:end-1])
                    sₕ = state_index(mdp, state)
                    aₕ = action_index(mdp, trajectory.action_hist[h])

                    # let sₕ, aₕ be current state, action
                    dl_dθₖ = β * ( dQ[sₕ,aₕ] - sum( [ lh(πᵦ,sₕ,ai⁻) * dQ[sₕ,ai⁻] for ai⁻ in actions_i ] ) )
                    ∇𝓛[k] += dl_dθₖ
                end
            end
        end
        θ += ∇𝓛
        # ∇Q = cal∇Q(ϕ)
    end
    EVD
end

lh(πᵦ, s,a) = πᵦ[s,a]

mdp, policy = generate_gridworld(10,10,γ=0.9)
trajectories = generate_trajectories(mdp, policy, 50)
ϕ = eye(100)
τ = 0.1

transition(mdp, GridWorldState(1,1), :up)

Pₐ = a2transition(mdp, :up)

state_index(mdp, GridWorldState(4,2))

EVD = MLIRL(mdp, ϕ, trajectories, 50, τ; ground_policy = policy)

using Plots
Plots.plot(EVD)

# i2s ✓
# π2transition ✓
