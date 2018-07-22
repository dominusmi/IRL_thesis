using JLD

results_folder = "/home/edoardo/Documents/Masters Work/Dissertation/results/DPM_BIRL"

results = load("$results_folder/4_120.jld")["results"]

function get_EVD_matrix(results, confidence)
	for key in keys(results)
		if key == confidence
			return results[confidence][2]
		end
	end
end

function get_log(results, confidence)
	for key in keys(results)
		if key == confidence
			return results[confidence][1]
		end
	end
end

function final_number_partitions(results, confidence)
	for key in keys(results)
		if key == confidence
			return size(results[confidence][2],1)
		end
	end
end
