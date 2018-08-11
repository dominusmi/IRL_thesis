# include("DiagGridWorld.jl")
using DiscreteValueIteration

size_x = 10
size_y = 10
states, terminals = DPMBIRL.generate_reward_states(size_x, size_y)

all_states = reshape([GridWorldState(x,y) for x in 1:size_x, y in 1:size_y], size_x*size_y)
rewards = reshape([states[y,x] for x in 1:size_x, y in 1:size_y], size_x*size_y)

mdp = DiagGridWorld(size_x,                                                                 # size x
				size_y,                                                                 # size y
				# map(x->GridWorldState(x[1:2]...), reward_states),                       # Reward states
				# map(x->x[3], reward_states),                                            # Respective rewards
				all_states,
				rewards,
				-1.0,                                                       # Boundary penalty
				1.0,                                                            # Transition probability
				Set([GridWorldState(x[1:2]...) for x in terminals]),      	# Terminal states
				0.9                                                                       # Discount factor γ
			)


solver = ValueIterationSolver(max_iterations=100, belres=1e-3)
policy = ValueIterationPolicy(mdp)
policy = solve(solver, mdp, policy)

heatmap(reshape(policy.util[1:end-1], (10,10)))
