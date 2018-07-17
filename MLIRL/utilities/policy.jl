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

            # Boltzmann
            # πᵦ = exp.(π.qmat[s,:]) / sum(exp.(π.qmat[s,:]))
            # ϵ-greedy
            # ϵ = 0.05
            # πᵦ = ones(4)*ϵ/3
            # πᵦ[indmax(π.qmat[s,:])] = 1-ϵ
            # Greedy
            πᵦ = zeros(4)
            πᵦ[indmax(π.qmat[s,:])] = 1.

            # for each possible action
            nbs = state_neighbours(mdp,s)
            for action in actions_space

                a = action_index(mdp, action)

                ∑s = 0.
                # Given an action, for each possible neighbour
                for (i,nb) in enumerate(state_neighbours(mdp, s))
                    r, Pₛₛ = 0., 0.
                    # if nb == s
                    #     r = -1.
                    # else
                    #     GridWorldState(ind2sub(sizes, nb)...)
                    #     index = find(mdp.reward_states .== nb_state)
                    #
                    #     if !isempty(index)
                    #         r = mdp.reward_values[index[1]]
                    #     end
                    # end

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
