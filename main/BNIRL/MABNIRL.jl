using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox

# TODO: replace globals by params
function MABNIRL(mdp, trajectories, η, κ; seed=1, max_iter=5e4, burn_in=500, use_assignements=true, ground_truth=nothing, punishment=5, use_clusters=false, n_goals=0)
	srand(seed)

	if !use_assignements
		@assert ground_truth !== nothing "Ground truth not given"
	end

	n_states 			= size(states(mdp),1)-1
	n_actions 			= size(actions(mdp),1)
	n_observations 		= size(trajectories,1)
	support_space 		= getSupportSpace(trajectories)
	n_support_states 	= size(support_space,1)
	ψ					= punishment
	n_trajectories		= size(trajectories,1)

	println("Punishment: $ψ")

	### Precompute all Q-values and their πᵦ
	tmp_dict, tmp_array, utils = precomputeQ(mdp, support_space)
	state2goal = tmp_dict
	all_goals  = tmp_array

	# Setup general variables
	const glb = Globals(n_states, n_actions, support_space,
						n_support_states, ψ, state2goal, all_goals, η, κ)


	clusters = Clusters(n_trajectories,
						fill(1, n_trajectories),
						collect(1:n_trajectories),
						Vector{Vector{Goal}}(0),
						zeros(Integer, n_trajectories),
						collect(1:n_trajectories))

	# TODO: rewrite this for multi-agent
	if use_assignements
		# Begin with one cluster per trajectory
		for i in 1:n_trajectories
			n_obs = size(trajectories[i],1)
			push!(clusters.G, [sample(Goal, glb) for i in 1:3])
			clusters.Z[i] = rand([1,2,3], n_obs)
		end
	else
		z = ground_truth
		n_goals = size(unique(z),1)
		goals =  [sample(Goal, glb) for i in 1:n_goals]
	end

	_log = Dict(:z=>[], :goals=>[])
	goal_hist = zeros(Integer, max_iter, 3)
	for t in 1:max_iter
		t%100 == 0 ? println("Iteration $t") : nothing


		# ╔═╗╦  ╦ ╦╔═╗╔╦╗╔═╗╦═╗╦╔╗╔╔═╗
		# ║  ║  ║ ║╚═╗ ║ ║╣ ╠╦╝║║║║║ ╦
		# ╚═╝╩═╝╚═╝╚═╝ ╩ ╚═╝╩╚═╩╝╚╝╚═╝
		for m in 1:n_trajectories
			update_cluster!(clusters, m)
		end

		for cₘ in 1:clusters.N

			assigned_to_cluster = find(clusters.assignements.==cₘ)

			# ╔═╗╔═╗╔═╗╦╔═╗╔╗╔  ╔═╗╔═╗╔═╗╦  ╔═╗
			# ╠═╣╚═╗╚═╗║║ ╦║║║  ║ ╦║ ║╠═╣║  ╚═╗
			# ╩ ╩╚═╝╚═╝╩╚═╝╝╚╝  ╚═╝╚═╝╩ ╩╩═╝╚═╝
			goals = clusters.G[c]
			zs = clusters.Z[assigned_to_cluster]
			for (i,curr_goal) in enumerate(goals)
				# Finds observations from all the cluster trajectories
				# which are assigned to "curr_goal"
				goal_observations = []
				for (j,traj) in enumerate(trajectories[assigned_to_cluster])
					tmp_indeces = zs[j] .== i
					vcat(goal_observations, traj[tmp_indeces])
				end
				# Sample goal
				goals[i] = resample(goals, goal_observations, glb)
			end


			# ╔═╗╔═╗╔═╗╦╔═╗╔╗╔  ╔═╗╔╗ ╔═╗╔═╗╦═╗╦  ╦╔═╗╔╦╗╦╔═╗╔╗╔╔═╗
			# ╠═╣╚═╗╚═╗║║ ╦║║║  ║ ║╠╩╗╚═╗║╣ ╠╦╝╚╗╔╝╠═╣ ║ ║║ ║║║║╚═╗
			# ╩ ╩╚═╝╚═╝╩╚═╝╝╚╝  ╚═╝╚═╝╚═╝╚═╝╩╚═ ╚╝ ╩ ╩ ╩ ╩╚═╝╝╚╝╚═╝
			# Re-assign observations
			for rep in 1:5
				for (i,z) in enumerate(zs)
					# Get the trajectory corresponding to the assignement vector
					observations = trajectories[ assigned_to_cluster[i] ]
					# Loop over observations
					for (i,obs) in enumerate(observations)
						reassign!(obs, i, z, goals, glb, use_clusters=use_clusters)
						postprocess!(z, goals)
					end
				end
			end
		end

		if t>burn_in
			push!(_log[:clusters], copy(clusters))
		end
	end
	_log, glb
end
