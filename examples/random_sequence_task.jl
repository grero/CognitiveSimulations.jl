
module RandomSequenceAnalysis
using RNNTrialStructures
using RecurrentNetworkModels
using RecurrentNetworkModels: Lux
using StableRNGs
using StatsBase
using Random
using JLD2
using Makie
using MultiDimensionalTimeSeriesPlots
using LinearAlgebra

include("training.jl")
include("subspace.jl")

function run_analysis()
    # set up training and testing trial structures
    apref = RNNTrialStructures.AngularPreference(collect(range(0.0f0, stop=2.0f0*π, length=17)), 5.0f0, 0.8f0);
    trialstruct = RNNTrialStructures.RandomSequenceTrial(20.0f0, 0.0f0, 20.0f0, 20.0f0, 2, 9, 16, apref);
    trial_iterator = RNNTrialStructures.generate_trials(trialstruct, 10_000, 20.0f0; rseed=UInt32(3),
                                                        pre_cue_multiplier=1.0f0, post_cue_multiplier=1.0f0, σ=0.0f0,
                                                        rng=StableRNG(1234));

    trialstruct_test = RNNTrialStructures.RandomSequenceTrial(20.0f0, 0.0f0, 20.0f0, 20.0f0, 9, 9, 16, apref);
    trial_iterator_test = RNNTrialStructures.generate_trials(trialstruct_test, 10_000, 20.0f0; rseed=UInt32(3),
                                                        pre_cue_multiplier=1.0f0, post_cue_multiplier=1.0f0, σ=0.0f0,
                                                        rng=StableRNG(1234));

    # load the model 
    (ps,st),model = train_model(trial_iterator, 400;nepochs=2_000, performance_aggregator=mean, accuracy_threshold=0.99f0,
                                                     output_nonlinearity=RecurrentNetworkModels.scaled_tanh, τ=1.0f0,η=0.1f0, load_only=true)


    xt,yt,wt = trial_iterator_test()

    # run the model on the test set
    (ŷ,h), ps2 = model(xt, ps, RecurrentNetworkModels.Lux.testmode(st))

    # get the error per sequence item
    err = RNNTrialStructures.compute_error(trialstruct_test, ŷ, yt)

    # chance error
    err0 = sin(2π/trialstruct.num_angles)

    idx0 = RNNTrialStructures.get_go_cue_onset(trialstruct_test, 9, 20.0f0)
    # get angles
    θ = permutedims(dropdims(mapslices(x->RNNTrialStructures.readout(trialstruct_test, x), yt[:,idx0:end,:], dims=1),dims=1))
    θp = permutedims(dropdims(mapslices(x->RNNTrialStructures.readout(trialstruct_test, x), ŷ[:,idx0:end,:], dims=1),dims=1))
    
    #compute cross error
    cross_error = zeros(Float32, size(θ,2), size(θ,2))
    for i2 in axes(cross_error,2) 
        for i1 in axes(cross_error,1)
            cross_error[i1,i2] = mean(sqrt.(sin.(θp[:,i1] .- θ[:,i2]).^2))
        end
    end

    θu = unique(θ)
    sort!(θu)

    # create subspace, one for each item
    Zp = Vector{Array{Float32,3}}(undef, 9)
    proj = Vector{Matrix{Float32}}(undef, 9)
    for k in 1:9
        _,pca = MultiDimensionalTimeSeriesPlots.mpca(h[:,idx0,:], θ[:,k])
        Z = mapslices(_x->predict(pca, _x), h, dims=1);
        Zp[k] = zeros(Float32, size(Z,1), size(Z,2), length(θu))
        for (i,_θ) in enumerate(θu)
            tidx = θ[:,k] .== _θ
            Zp[k][:,:,i] = dropdims(mean(Z[:,:,tidx],dims=3),dims=3)
        end
        proj[k] = pca.proj
    end
    
    #compute principal angle between subspaces
    pangle = fill(Float32(NaN), 9,9)
    for i1 in 1:9
        for i2 in i1:9
            ss = svd(proj[i1]'*proj[i2])
            pangle[i1,i2] = ss.S[1]
        end
    end

    # compute cross_error


    # plot
    label_padding = (10.0, 0.0, 0.0, 0.0)
    with_theme(theme_minimal()) do
        fig = Figure()
        lg1 = GridLayout(fig[1,1])
        ax = Axis(lg1[1,1])
        Label(lg1[1,1,TopLeft()], "A", padding=label_padding)
        xx = repeat(1:9, 1, size(err,2))
        boxplot!(ax, xx[:], err[2:end,:])
        ax.ylabel = "Error"
        ax.xticks = (1:9, string.(1:9))
        hlines!(ax, err0, color=:black, linestyle=:dot)

        ax2 = Axis(lg1[1,2])
        Label(lg1[1,2, TopLeft()], "B", padding=label_padding)
        h1 = heatmap!(ax2, cross_error)
        Colorbar(lg1[1,3], h1, label="Error")
        ax2.xticks = (1:9, string.(1:9))
        ax2.yticks = (1:9, string.(1:9))

        ax3 = Axis(lg1[1,4])
        Label(lg1[1,4, TopLeft()], "C", padding=label_padding)
        h = heatmap!(ax3, pangle)
        Colorbar(lg1[1,5],h, label="Principal angle")
        ax3.xticks = (1:9, string.(1:9))
        ax3.yticks = (1:9, string.(1:9))

        # show subspace for the first 4 items
        lg = GridLayout(fig[2,1])
        Label(fig[2,1,TopLeft()], "D", padding=label_padding)
        for j in 1:4
            ax = Axis3(lg[1,j],xgridvisible=true, ygridvisible=true, zgridvisible=true,
                               xticklabelsvisible=true, yticklabelsvisible=false,
                               zticklabelsvisible=false, xlabelvisible=false,
                               ylabelvisible=false, zlabelvisible=false)#, viewmode=:stretch)
            ax.title = "s$j"
            MultiDimensionalTimeSeriesPlots.plot_network_trials!(ax, Zp[j], θ[:,j:j])
        end
        rowsize!(fig.layout, 1, Relative(0.5))
        fig
    end
end
end # module