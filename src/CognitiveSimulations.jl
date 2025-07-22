module CognitiveSimulations
using RecurrentNetworkModels
using RNNTrialStructures
using JLD2
using StableRNGs
using StatsBase
using MultiDimensionalTimeSeriesPlots
using Glob

"""
    get_subspace(X::AbstractVector{T,3}, θ::Matrix{T2}, idx0::Int64) where T <: Real where T2

Find an orthongal subspace containing information about θ at time point `idx0`.
"""
function get_subspace(X::AbstractArray{T,3}, θ::Matrix{T2}, idx0::Int64) where T <: Real where T2
    Y = dropdims(mean(X[:,idx0:idx0+1, :],dims=2),dims=2)   
    _,w12 = MultiDimensionalTimeSeriesPlots.rpca(Y, MultiDimensionalTimeSeriesPlots.Angle.(θ[:,1]), MultiDimensionalTimeSeriesPlots.Angle.(θ[:,2]))
    Z12 = mapslices(x->w12*x, X, dims=1)
    Z12, w12
end

function load_model(fname::String)
    ps,st = JLD2.load(fname, "params","state")
    pf = split(fname, '_')
    insert!(pf, length(pf), "args")
    params_file = join(pf, '_')
    args = JLD2.load(params_file)
    pp = splitpath(fname)
    # get the directory name
    dname = joinpath(pp[1:end-1])
    # load the trial structure 
    trialstruct = JLD2.load(joinpath(dname, "trialstruct.jld2"), "trialstruct")
    # load the parameters used to train the model
    if haskey(args, "h0")
        h0 = args["h0"]
        ptname = joinpath(dname, "trial_iterator_$(string(h0, base=16)).jld2")
        trials_args = JLD2.load(ptname,"args")
    else
        trials_args = Dict()
    end
    ps, st, trialstruct, args, trials_args
end

function find_model(trialstruct;kwargs...)
    dargs = Dict(kwargs)
    # find the directory for the requested trialstruct
    task_name = RNNTrialStructures.get_name(trialstruct)
    task_signature = RNNTrialStructures.signature(trialstruct)
    dname = joinpath(@__DIR__, "..", "data", "$(task_name)_$(string(task_signature,base=16))")
    # find all trial_iterator files
    args_file = "trial_iterator_*.jld2"
    cd(dname) do
        mpfiles = glob("model_state_args_*.jld2")
        good_mpfiles = String[]
        trial_iterator_signs = UInt32[]
        for mf in mpfiles
            keep = true 
            args = JLD2.load(mf)
            for (k,v) in args
                if k in keys(dargs)
                    if v != dargs[k]
                        keep = false
                        break
                    end
                end
            end
            if keep
                if "h0" in keys(args)
                    push!(trial_iterator_signs, args["h0"])
                else
                    push!(trial_iterator_signs, UInt32(0))
                end
                push!(good_mpfiles, mf)
            end
        end

        arg_files = glob(args_file)
        model_files = String[]
        trial_files = String[]
        for af in arg_files
            # check the hash
            keep = false 
            ii = 0
            for (jj,sig) in enumerate(trial_iterator_signs)
                ss = string(sig, base=16)
                if occursin(ss, af)
                    keep = true
                    ii = jj
                    break
                end
            end
            if keep
                args = JLD2.load(af, "args")
                for (k,v) in args
                    if k in keys(dargs)
                        if v != dargs[k]
                            keep = false
                            break
                        end
                    end
                end
            end
            if keep
                push!(trial_files, joinpath(dname,af))
                push!(model_files, joinpath(dname,good_mpfiles[ii]))
            end
        end
        model_files, trial_files
    end
    # find all model parameter files
end

function train_model(trialstruct, nhidden::Int64;batchsize=256, randomize_go_cue=false, σ=0.0316f0, post_cue_multiplier=2.0f0, rseed=12335, nepochs=20_000, accuracy_threshold=0.95f0,
        learning_rate=Float32(1e-4), redo=false, go_cue_onset_min::Float32=zero(Float32), go_cue_onset_max::Float32=go_cue_onset_min, stim_onset_min::Vector{Float32}=zeros(Float32,
        trialstruct.nangles), stim_onset_max=stim_onset_min, load_only=false)

    rng = StableRNG(rseed)
    task_name = RNNTrialStructures.get_name(trialstruct)
    task_signature = RNNTrialStructures.signature(trialstruct)
    dname = joinpath(@__DIR__, "..", "data", "$(task_name)_$(string(task_signature,base=16))")
    if !isdir(dname)
        mkdir(dname)
    end
    ninputs = RNNTrialStructures.num_inputs(trialstruct)
    noutputs = RNNTrialStructures.num_outputs(trialstruct)

    model = RecurrentNetworkModels.LeakyRNNModel(ninputs, nhidden, noutputs)
    trial_iterator = RNNTrialStructures.generate_trials(trialstruct, batchsize;randomize_go_cue=randomize_go_cue,
                                                        σ=σ, go_cue_onset_min=go_cue_onset_min, go_cue_onset_max=go_cue_onset_max,
                                                        post_cue_multiplier=post_cue_multiplier,
                                                        stim_onset_min=stim_onset_min, stim_onset_max=stim_onset_max,
                                                        rng=rng, rseed=rseed)
    args_file = "trial_iterator_$(string(trial_iterator.arghash, base=16)).jld2"
    args = Dict(:batchsize => batchsize,
                :randomize_go_cue => randomize_go_cue,
                :σ => σ,
                :post_cue_multiplier => post_cue_multiplier,
                :rng => rng,
                :rseed => rseed,
                :go_cue_onset_min => go_cue_onset_min,
                :go_cue_onset_max => go_cue_onset_max,
                :stim_onset_min => stim_onset_min,
                :stim_onset_max => stim_onset_max)

    cd(dname) do
        # save only once since all models in this folder will use the same trial structure
        if !isfile("trialstruct.jld2")
            JLD2.save("trialstruct.jld2", Dict("trialstruct" => trialstruct))
        end
        JLD2.save(args_file, Dict("args"=>args))
        compute_acc(ŷ, y) = mean(RNNTrialStructures.performance(trialstruct, ŷ,y))
        compute_perf(ŷ, y) = mean(RNNTrialStructures.performance(trialstruct, ŷ,y;require_fixation=false))

        ps = RecurrentNetworkModels.train_model(model, trial_iterator, compute_acc, compute_perf;nepochs=nepochs,redo=redo,
                                                                                          learning_rate=learning_rate, accuracy_threshold=accuracy_threshold,
                                                                                          save_file="model_state.jld2",h=trial_iterator.arghash,rseed=rseed,
                                                                                          load_only=load_only)
        return ps, model, trial_iterator
    end
end

end # module


