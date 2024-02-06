import Pkg
Pkg.add("Revise")
Pkg.add(path="../")

using Test
using DelimitedFiles
using Statistics
using openBF

samples = parse(Int64,ARGS[1])
network_names = ARGS[2:end]
print(network_names)
path = pwd()

for network_name in network_names
	cd(network_name)
	for i in 1:samples
		openBF.runSimulation(network_name*".yml", verbose=true)
	end
	cd("../")
end

#@testset "openBF.jl" begin
#
#    #unit tests
#    println("Test initialise.jl functions")
#    include("test_initialise.jl")
#
#    println("Test boundary_conditions.jl functions")
#    include("test_boundary_conditions.jl")
#
#    #integration tests
#    println("Test networks")
#    include("test_single-artery.jl")
#    include("test_conjunction.jl")
#    include("test_bifurcation.jl")
#    include("test_aspirator.jl")
#    include("test_tapering.jl")
#end
