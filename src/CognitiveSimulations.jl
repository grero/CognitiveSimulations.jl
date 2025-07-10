module CognitiveSimulations
using RecurrentNetworkModels
using RNNTrialStructures
using JLD2
using StableRNGs
using StatsBase

function train_model(trialstruct, nhidden::Int64;batchsize=256, randomize_go_cue=false, σ=0.0316f0, post_cue_multiplier=2.0f0, rseed=12335, nepochs=20_000, accuracy_threshold=0.95f0,
                    learning_rate=Float32(1e-4), redo=false, go_cue_onset_min::Float32=zero(Float32), go_cue_onset_max::Float32=go_cue_onset_min)

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
                                                        post_cue_multiplier=post_cue_multiplier, rng=rng, rseed=rseed)
    args = Dict(:batchsize => batchsize,
                :randomize_go_cue => randomize_go_cue,
                :σ => σ,
                :post_cue_multiplier => post_cue_multiplier,
                :rng => rng,
                :rseed => rseed,
                :go_cue_onset_min => go_cue_onset_min,
                :go_cue_onset_max => go_cue_onset_max)

    cd(dname) do
        # save only once since all models in this folder will use the same trial structure
        if !isfile("trialstruct.jld2")
            JLD2.save("trialstruct.jld2", Dict("trialstruct" => trialstruct, "args" => args))
        end
        ps = RecurrentNetworkModels.train_model(model, trial_iterator, (ŷ,y)->mean(RNNTrialStructures.matches(trialstruct, ŷ,y));nepochs=nepochs,redo=redo,
                                                                                          learning_rate=learning_rate, accuracy_threshold=accuracy_threshold,
                                                                                          save_file="model_state.jld2",h=RNNTrialStructures.signature(trialstruct),rseed=rseed)
        return ps, model, trial_iterator
    end
end

end # module


