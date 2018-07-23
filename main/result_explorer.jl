using JLD
using Plots

results_folder = "/home/edoardo/Documents/Masters Work/Dissertation/results/DPM_BIRL"
results_folder = "/run/user/1000/gvfs/sftp:host=godzilla.csc.warwick.ac.uk,user=phujsc/home/space/phujsc/Documents/main/results"

results = load("$results_folder/2_40.jld")
results = results["results"]

lhs = get_lhs(results, 1.0)

evds = get_EVD_matrix(results, 1.0)
lhs  = _log[:likelihoods]
evds_1 = zeros(350, 4)
evds_2 = zeros(350, 4)

lh_1 = zeros(350, 4)

for (i,evd) in enumerate(get_EVD_matrix(results, 1.0))
	for j in 1:4
		if j <= size(evd,2)
			evds_1[i,j] = evd[1,j]
			evds_2[i,j] = evd[2,j]
			lh_1[i,j] = lhs[i][j]
		end
	end
end

fig = Plots.plot(evds_1[:,1])
Plots.plot!(lh_1[:,1]/maximum(lh_1[:,1])*20)
savefig(fig, "MCMC 400 iterations")


function get_EVD_matrix(results, confidence)
	for key in keys(results)
		if key == confidence
			return results[confidence][:EVDs]
		end
	end
end

function get_lhs(results, confidence)
	for key in keys(results)
		if key == confidence
			return results[confidence][:likelihoods]
		end
	end
end

function get_log(results, confidence)
	for key in keys(results)
		if key == confidence
			return results[confidence]
		end
	end
end

function final_number_partitions(results, confidence)
	for key in keys(results)
		if key == confidence
			return size(results[confidence][:EVDs],1)
		end
	end
end
