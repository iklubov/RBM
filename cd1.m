function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    hiddenProb = visible_state_to_hidden_probabilities(rbm_w, visible_data)
    hiddenState = sample_bernoulli(hiddenProb)
    gradient1 = configuration_goodness_gradient(visible_data, hiddenState)
    visibleProb = hidden_state_to_visible_probabilities(rbm_w, hiddenState)
    visibleStateReconstr = sample_bernoulli(visibleProb)
    hiddenProbReconstr = visible_state_to_hidden_probabilities(rbm_w, visibleStateReconstr)
    hiddenStateReconstr = sample_bernoulli(hiddenProbReconstr)
    gradient2 = configuration_goodness_gradient(visibleStateReconstr, hiddenStateReconstr)
    ret = gradient1 - gradient2
end
