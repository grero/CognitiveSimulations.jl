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
    hd_step = trials_args[:Δθstep]
    kwargs = NamedTuple{filter(k->!in([:ntrials, :trialstruct, :dt, :Δθstep])(k), keys(trials_args))}(trials_args)
    #hack

    @show keys(kwargs)
    trial_iterator = RNNTrialStructures.generate_trials(trialstruct, trials_args.ntrials, trials_args.dt;hd_step=hd_step, kwargs...)
    n_hh, n_in = size(ps.rnn_cell.weight_ih)
    n_out = RNNTrialStructures.num_outputs(trialstruct)
    model = RecurrentNetworkModels.LeakyRNNModel(n_in, n_hh, n_out)
    model, ps, st, args, trial_iterator 
end

function find_model(trial_iterator::RNNTrialStructures.TrialIterator;kwargs...)
    trialstruct = trial_iterator.args.trialstruct
    task_name = RNNTrialStructures.get_name(trialstruct)
    task_signature = RNNTrialStructures.signature(trialstruct)
    dname = joinpath(@__DIR__, "..", "data", "$(task_name)_$(string(task_signature,base=16))")
    model_files = String[]
    cd(dname) do
        mpfiles = glob("model_state_args_*.jld2")
        good_mpfiles = String[]
         for mf in mpfiles
            keep = true 
            args = JLD2.load(mf)
            if "h0" in keys(args)
                if args["h0"] == trial_iterator.arghash
                    push!(model_files, joinpath(dname,mf))
                end

            end
        end
    end
    model_files
end

function find_model(trialstruct::RNNTrialStructures.AbstractTrialStruct;datadir=joinpath(@__DIR__, "..", "data"), kwargs...)
    dargs = Dict(kwargs)
    # find the directory for the requested trialstruct
    task_name = RNNTrialStructures.get_name(trialstruct)
    task_signature = RNNTrialStructures.signature(trialstruct)
    dname = joinpath(datadir, "$(task_name)_$(string(task_signature,base=16))")
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
                for k in keys(args)
                    v = args[k]
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

function train_model(trial_iterator::RNNTrialStructures.TrialIterator, nhidden::Int64; rseed::Union{UInt32,Nothing}=nothing, rng::Union{AbstractRNG, Nothing}=nothing, nepochs=20_000, accuracy_threshold=0.95f0,
        learning_rate=Float32(1e-4), redo=false,performance_aggregator::Function = mean, output_nonlinearity=Lux.sigmoid, τ=0.2f0, η=0.0f0, load_only=false)

    if rseed === nothing
        if haskey(trial_iterator.args, :rseed)
            rseed = trial_iterator.args.rseed
        else
            rseed = UInt32(1234)
        end
    end
    if rng === nothing
        if haskey(trial_iterator.args, :rng)
            rng = trial_iterator.args.rng
        else
            rng = StableRNG(rseed)
        end
    end
    trialstruct = trial_iterator.args.trialstruct
    task_name = RNNTrialStructures.get_name(trialstruct)
    task_signature = RNNTrialStructures.signature(trialstruct)
    dname = joinpath(@__DIR__, "..", "data", "$(task_name)_$(string(task_signature,base=16))")
    if !isdir(dname)
        mkdir(dname)
    end
    ninputs = RNNTrialStructures.num_inputs(trialstruct)
    noutputs = RNNTrialStructures.num_outputs(trialstruct)

    model = RecurrentNetworkModels.LeakyRNNModel(ninputs, nhidden, noutputs;output_nonlinearity=output_nonlinearity,τ=τ,η=η)
    args_file = "trial_iterator_$(string(trial_iterator.arghash, base=16)).jld2"

    cd(dname) do
        # save only once since all models in this folder will use the same trial structure
        if !isfile("trialstruct.jld2")
            JLD2.save("trialstruct.jld2", Dict("trialstruct" => trialstruct))
        end
        JLD2.save(args_file, Dict("args"=>trial_iterator.args))
        compute_acc(ŷ, y) = performance_aggregator(RNNTrialStructures.performance(trialstruct, ŷ,y))
        compute_perf(ŷ, y) = performance_aggregator(RNNTrialStructures.performance(trialstruct, ŷ,y;require_fixation=false))

        ps = RecurrentNetworkModels.train_model(model, trial_iterator, compute_acc, compute_perf;nepochs=nepochs,redo=redo,
                                                                                          learning_rate=learning_rate, accuracy_threshold=accuracy_threshold,
                                                                                          save_file="model_state.jld2",h=trial_iterator.arghash,rseed=rseed,
                                                                                          load_only=load_only)
        return ps, model
    end
end
