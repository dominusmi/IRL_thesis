using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox

### Precompute all Q-values and their πᵦ

### Sample initial subgoal

### Initialise assignements zᵢ

### Main loop

for i in 1:max_iter
	for gⱼ in G
		# Calculate likelihood for all possible subgoals

		# Set multinomial probability as P(oᵢ|gⱼ) x P(gⱼ)

		# Sample from multinomial
	end

	for oᵢ in O
		for g in G
			# j: cluster of g
			### Calculate p(zᵢ=j|z,O,Rⱼ)
			# P(zᵢ|z₋ᵢ) = #observations in zᵢ / n-1+η
			# P(oᵢ|g_zᵢ) = likelihood of observation given assignement
		end
		# calculate probability of new partition

		# normalise and multinomial sample
	end
end
