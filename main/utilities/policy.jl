"""
    Find the value function for a given policy

    η is meant to replace the unknown Pₛₛ transition probability we do not know
    We're assuming not moving is not a possible action except if out of bounds
"""
function policy_evaluation(mdp::MDP, π::Policy, ϵ=0.01::Float64; η=0.9)
    n_states = size(states(mdp),1)-1
    V = zeros(n_states)
    actions_space = actions(mdp)

    sizes = (mdp.size_x, mdp.size_y)
    n_iters = 0
    Δ = 10
    while Δ > ϵ
        Δ = 0
        # For each state
        for s in 1:n_states
            v = V[s]
            ∑π = 0.
            gw_state = GridWorldState(ind2sub(sizes, s)...)

            #### NOTE ####
            # Tested all three policies over a single example, it appears
            # that boltzmann definitely does not work well, with EVD
            # increasing, and ϵ-greedy and greedy perform very similarly
            # and require more tests to check if one is better than the other

            ####    Boltzmann   ####
            # πᵦ = exp.(π.qmat[s,:]) / sum(exp.(π.qmat[s,:]))
            ####    ϵ-greedy    ####
            πᵦ = ones(4)*ϵ/3
            πᵦ[indmax(π.qmat[s,:])] = 1-ϵ
            #####   Greedy      ####
            # πᵦ = zeros(4)
            # πᵦ[indmax(π.qmat[s,:])] = 1.

            # for each possible action
            nbs = state_neighbours(mdp,s)
            for action in actions_space

                a = action_index(mdp, action)

                ∑s = 0.
                # Given an action, for each possible neighbour
                for (i,nb) in enumerate(state_neighbours(mdp, s))
                    r, Pₛₛ = 0., 0.

                    r = reward(mdp, gw_state, action)

                    if i == a
                        Pₛₛ = η
                    else
                        Pₛₛ = (1-η)/(n_actions(mdp)-1)
                    end
                    ∑s += Pₛₛ*( r + mdp.discount_factor * V[nb] )
                end
                ∑π += πᵦ[a] * ∑s
            end
            V[s] = ∑π

            Δ = max(Δ, abs(v-V[s]))
            n_iters += 1
        end
    end
    V
end

"""
    Calculates Boltzmann policy given a Q-matrix and β (inverse of temperature)
"""
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

"""
    Returns the transition matrix of a policy
"""
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
