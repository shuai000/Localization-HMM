"""
1. Hidden Markov Based Module
2. Ref: Lawrence tutorial on HMM & selected application in speech recognition
3. Author: Shuai Sun    05/09/2018
-----------------------------------------------------------------------------
Parameter needs to be specified once a HMM model instance is initialized
"""
import numpy as np
import scipy.io as sio
import scipy.stats as stats
from math import log, exp
import matplotlib.pyplot as plt


class HiddenMarkov:

    def __init__(self, para_hmm, specification=False):
        """
        Called right after setting the HMM model
        Set the hidden Markov parameters
        :param para_hmm: dictionary format
        :return:
        """
        assert 'state_num' in para_hmm, " HMM state number is not provided!"
        self.Ns = para_hmm['state_num']
        self.Pi = np.ones(self.Ns) / self.Ns

        assert 'anchor_num' in para_hmm, " HMM anchor number is not provided!"
        self.Na = para_hmm['anchor_num']

        assert 'initial_p' in para_hmm, " HMM state initial probability is not provided!"
        self.Pi = para_hmm['initial_p']
        assert self.Pi.shape[0] == self.Ns, " Number of state doesn't match model initial probability size! "

        assert 'transition_p' in para_hmm, " HMM state transition probability is not provided!"
        self.Pt = para_hmm['transition_p']
        self.log_pt = np.zeros((self.Ns, self.Ns))
        for i in range(self.Ns):
            for j in range(self.Ns):
                if self.Pt[i, j] != 0:
                    self.log_pt[i, j] = log(self.Pt[i, j])
                else:
                    self.log_pt[i, j] = float('-inf')
        assert self.Pt.shape == (self.Ns, self.Ns), " Number of state doesn't match model transition probability"

        assert 'training_type' in para_hmm, " HMM training type is not provided!"
        # 1. from_loading 2. from_external 3. from_data
        training_from = para_hmm['training_type']
        if training_from == 'from_loading':
            self.training()  # load training parameter
        elif training_from == 'from_external':
            if 'mean' in para_hmm:
                self.mean = para_hmm['mean']
                assert self.mean.shape == (self.Na, self.Ns), \
                    " Number of state/observation don't match model mean size!"
            else:
                assert False, " NO mean parameter provided to the model!"
            if 'cov' in para_hmm:
                self.cov = para_hmm['cov']
                assert self.cov.shape == (self.Na, self.Na, self.Ns), \
                    "Number of state/observation don't match model cov size!"
            else:
                assert False, "NO COV parameter provided to the model!"
        elif training_from == 'not_provided':
            pass
        else:
            raise Exception('training_type is in wrong format: from_loading, from_external or not_provided!')

        assert 'likelihood_from' in para_hmm, "HMM observation filename is not provided!"
        if para_hmm['likelihood_from'] == 'from_external':  # filename in None means likelihood is provided outside
            assert 'likelihood' in para_hmm, "HMM observation likelihood is not provided"
            self.likelihood = para_hmm['likelihood']
            assert self.likelihood.shape[1] == self.Ns, "Number of state doesn't match likelihood data size"
        elif para_hmm['likelihood_from'] == 'from_file':
            assert 'filename' in para_hmm, "HMM evaluation data filename is not provided!"
            data, status = self.data_extraction(para_hmm['filename'], 0)
            self.likelihood = self.get_likelihood(data, status)
        elif para_hmm['likelihood_from'] == 'not_provided':
            pass
        else:
            raise Exception("Likelihood from is not in the library!")

        if specification:  # Print out the HMM model set up in the console
            self.model_specification(para_hmm)

    def set_likelihood(self, likelihood_value):
        self.likelihood = likelihood_value
        assert self.likelihood.shape[1] == self.Ns, "Number of state doesn't match likelihood data size"

    @staticmethod
    def state_estimation(alpha):
        st = alpha.argmax(axis=1)
        st = st[:] + 1  # physically map to state index
        return st

    def set_gaussin(self, mu, cov):
        self.mean = mu
        self.cov = cov

        assert self.mean.shape == (self.Na, self.Ns), "Number of state and observation don't match model mean size!"
        assert self.cov.shape == (self.Na, self.Na, self.Ns), "Number of state/observation don't match model cov size!"

    def model_specification(self, para):
        print("\n\n-------- HIDDEN MARKOV MODEL ---------")
        print("State num: {}".format(self.Ns))
        if para['training_type'] == 'not_provided':
            print("Training data is not specified initially, provided later!")
        else:
            print("Training data is from: {}".format(para['training_type']))

        if para['likelihood_from'] == 'not_provided':
            print("Likelihood data is not specified initially, provided later!")
        else:
            print("Likelihood is from: {}".format(para['likelihood_from']))
        print("-------- HIDDEN MARKOV MODEL ---------\n")

    def forward_process(self, p_type='none', interval_index=None, prior=None):
        """
        Hidden Markov Model standard forward process
        func: Recursively compute forward variable
              alpha_t(i) = p(O_[1:t], q_t = S_i|lambda)
        :param p_type: propagation type: 'none', 'posterior', 'logform' (logform is not used, has to be approximated)
        :return: forward variable and estimation based on purely forward process
        """
        if interval_index is None:
            like_value = self.likelihood
        else:
            assert len(interval_index) == 2, "interval format wrong!"
            assert interval_index[1] > interval_index[0], "interval format wrong!"
            like_value = self.likelihood[interval_index[0]:interval_index[1] + 1, :]
        alpha = np.zeros((like_value.shape[0], self.Ns))

        # Forward Algorithm
        if p_type == 'none':
            for i_time in range(like_value.shape[0]):
                if i_time == 0:  # initialization
                    alpha[i_time, :] = self.Pi * like_value[i_time, :]  # Eq.(19)
                else:  # induction
                    alpha[i_time, :] = self.forward_propagation(alpha[i_time - 1, :]) \
                                       * like_value[i_time, :]  # Eq. (20)
            s_forward = self.state_estimation(alpha)
            return s_forward, alpha
        elif p_type == 'posterior':
            ct = np.zeros(like_value.shape[0])
            # initialization, if no pripr provided, then it is assumed as uniform
            if prior is None:
                alpha[0, :] = self.Pi * like_value[0, :]  # Eq.(19)
            else:
                alpha[0, :] = prior * like_value[0, :]
            ct[0] = sum(alpha[0, :])
            alpha[0, :] = alpha[0, :] / ct[0]
            # induction
            for i_time in range(1, like_value.shape[0]):
                alpha[i_time, :] = self.forward_propagation(alpha[i_time - 1, :]) \
                                   * like_value[i_time, :]  # Eq. (20)
                ct[i_time] = sum(alpha[i_time, :])
                alpha[i_time, :] = alpha[i_time, :] / ct[i_time]  # scaling p(si,o_1:t) -> p(si|o_1:t)
            log_likelihood = sum(np.log(ct))
            s_forward = self.state_estimation(alpha)
            return s_forward, alpha, log_likelihood, ct
        elif p_type == 'logform':
            for i_time in range(like_value.shape[0]):
                if i_time == 0:  # initialization
                    for i in range(self.Ns):
                        alpha[i_time, i] = np.log(self.Pi[i]) + np.log(like_value[i_time, i])
                else:
                    alpha[i_time, :] = self.forward_propagation(alpha[i_time - 1, :], p_type='log')
                    for i in range(self.Ns):
                        alpha[i_time, i] = alpha[i_time, i] + np.log(like_value[i_time, i])
            log_likelihood = 0
            for i in range(self.Ns):
                log_likelihood = log_likelihood + exp(alpha[-1, i])
            log_likelihood = log(log_likelihood)
            s_forward = self.state_estimation(alpha)
            return s_forward, alpha, log_likelihood

    def forward_propagation(self, alpha_p, p_type='none'):

        if p_type == 'none':
            alpha_n = np.sum(alpha_p * self.Pt.T, axis=1)
        elif p_type == 'log':
            alpha_n = np.zeros(len(alpha_p))
            for j in range(self.Ns):
                for i in range(self.Ns):
                    alpha_n[j] = alpha_n[j] + exp(alpha_p[i] + np.log(self.Pt[i, j]))
                alpha_n[j] = np.log(alpha_n[j])
        return alpha_n

    def backward_process(self, p_type='none', scaling_factor=None, interval_index=None):

        if interval_index is None:
            like_value = self.likelihood
        else:
            assert len(interval_index) == 2, "interval format wrong!"
            assert interval_index[1] > interval_index[0], "interval format wrong!"
            like_value = self.likelihood[interval_index[0]:interval_index[1] + 1, :]
        beta = np.zeros((like_value.shape[0], self.Ns))

        if p_type == 'none':
            # 1. initialization
            beta[-1, :] = 1
            # 2. induction (propagation back)
            for i_time in range(like_value.shape[0] - 2, -1, -1):
                beta[i_time, :] = self.back_propagation(beta[i_time + 1, :], like_value[i_time + 1, :])
        elif p_type == 'posterior':
            assert scaling_factor is not None, "The scaling fator is not provided!"
            beta[-1, :] = 1 / scaling_factor[-1]
            for i_time in range(like_value.shape[0] - 2, -1, -1):
                beta[i_time, :] = self.back_propagation(beta[i_time + 1, :], like_value[i_time + 1, :])
                beta[i_time, :] = beta[i_time, :] / scaling_factor[i_time]
        return beta

    def back_propagation(self, beta_p, like_value):
        beta_n = np.sum(beta_p * like_value * self.Pt, axis=1)
        return beta_n

    def forward_backward(self, p_type='posterior', window_length=None, e_position='middle'):
        """
        Forward-Backward algorithm for optimal state estimation
        This optimality criterion maximizes the expected number of correct individual states
        j = arg max_(i){y_t(i) = p(qt = Si | O, lambda)}
        :param filename:  filename of txt that contains RSS data
        :param window_length, the sliding window length for the batch processing based forward backward algorithm
                              if it is none, then the whole data sequence is processed
                              otherwise, it is window sliding based, move one step forward each time
                              -----.------
                               -----.------
        :param e_position, by default, 'middle', the middle one is output as current estimate within the windonw length
                                   'front', 'end' are the alternative
        :return: the estimated state sequence, st, as well as the posterior probability, yita
        """

        if window_length is None:  # process the whole data sequence
            if p_type == 'none':
                alpha = self.forward_process()[1]
                beta = self.backward_process()
            elif p_type == 'posterior':
                output = self.forward_process(p_type=p_type)
                alpha = output[1]
                ct = output[3]
                beta = self.backward_process(p_type=p_type, scaling_factor=ct)

            yita = alpha * beta
            st = self.state_estimation(yita)
            return st, yita
        else:  # sliding window based processing
            end_time = self.likelihood.shape[0]
            st = np.zeros(end_time)
            yita = np.zeros((end_time, self.Ns))

            if e_position == 'middle':
                e_index = int(window_length / 2)
            elif e_position == 'front':
                e_index = 0
            elif e_position == 'end':
                e_index = -1
            else:
                pass  # drop an error here

            assert window_length > 1, "window length can't be less than one!"
            window_length = window_length - 1  # python start from zero
            # initial step
            interval = [0, window_length]
            output = self.forward_process(p_type=p_type, interval_index=interval, prior=None)
            alpha = output[1]
            ct = output[3]
            beta = self.backward_process(p_type=p_type, scaling_factor=ct, interval_index=interval)

            yita_temp = alpha * beta
            st_temp = self.state_estimation(yita_temp)
            # assign from begiing to the middle point
            yita[0:e_index + 1, :] = yita_temp[0:e_index + 1, :]
            st[0:e_index + 1] = st_temp[0: e_index + 1]

            for i_time in range(1, end_time - window_length):
                interval = [i_time, i_time + window_length]
                output = self.forward_process(p_type=p_type, interval_index=interval, prior=yita_temp[0, :])
                alpha = output[1]
                ct = output[3]
                beta = self.backward_process(p_type=p_type, scaling_factor=ct, interval_index=interval)

                yita_temp = alpha * beta
                st_temp = self.state_estimation(yita_temp)

                if i_time == end_time - window_length - 1:
                    # the end of sliding window, take the estimate result of all
                    yita[i_time + e_index: end_time, :] = yita_temp[e_index:window_length + 1, :]
                    st[i_time + e_index: end_time] = st_temp[e_index:window_length + 1]
                else:  # normal sliding process
                    yita[i_time + e_index, :] = yita_temp[e_index, :]
                    st[i_time + e_index] = st_temp[e_index]
            return st, yita

    def likelihood_com(self, data, status):
        """
        Function: compute likelihood based on multivariate Gaussian distribution
        :param data: RSS for all anchors, note that
        :param status: indicator of which anchor is active, the probability evaluation
                       will change size based on status's configuration
        :return:  likelihood probability, normalized to one
        """
        likelihood = np.zeros(self.Ns)
        if data.any():
            for i in range(self.Ns):
                mean_i = self.mean[status, i]
                cov_i = self.cov[:, :, i][status][:, status]  # take the sub-covariance
                likelihood[i] = stats.multivariate_normal.pdf(
                    data, mean_i, cov_i)
            norm_sum = likelihood.sum()
            likelihood = likelihood / norm_sum
        else:
            likelihood[:] = 1 / self.Ns
        return likelihood

    def get_likelihood(self, data, status):

        likelihood = np.zeros((data.shape[0], self.Ns))

        for i_time in range(data.shape[0]):
            tempstatus = np.where(status[i_time, :] == 1)[0]
            tempdata = data[i_time, :][tempstatus]

            # Following is due to requirement from stats library
            # maybe not efficient
            reformat_data = np.zeros((1, len(tempstatus)))
            reformat_data[0, :] = tempdata

            likelihood[i_time, :] = self.likelihood_com(reformat_data, tempstatus)

        return likelihood

    def training(self, t_type='load', train_data=[]):
        if t_type == 'load':
            self.mean = sio.loadmat('data/mean_x.mat')['mean_x']
            self.cov = sio.loadmat('data/cov_x.mat')['cov_x']
        elif t_type == 'list':  # I don't understand the meaning of list nowadays
            rx_num = train_data[0].shape[1]
            state_num = len(train_data)
            train_mean = np.zeros((rx_num, state_num))
            train_cov = np.zeros((rx_num, rx_num, state_num))
            for idx, val in enumerate(train_data):
                tmp_mean = val.mean(axis=0)
                tmp_error = val - tmp_mean
                tmp_cov = np.dot(tmp_error.T, tmp_error) / (val.shape[0] - 1)
                train_mean[:, idx] = tmp_mean
                train_cov[:, :, idx] = tmp_cov
            return train_mean, train_cov

    def para_estimation(self, re_type='posterior', max_iternum=10):
        """
        Function: Estimate HMM parameter (Pt, Pi) using EM algorithm
        Ref: Rabiner tutorial paper  05/10/2018
        :param re_type: control the forward and backward process (scaled or not)
                        by default it is scaled
        :param max_iternum: the maximum iteration number used
        :return: estimated Pt, Pi, and log_likelihood sequence for each iteration
        IMPORTANT: please note that it automatically update self.Pt and self.Pi in the instance
        """
        if re_type == 'posterior':
            a, alpha, c, ct = self.forward_process(p_type=re_type)
            beta = self.backward_process(p_type=re_type, scaling_factor=ct)
            log_likelihood = sum(np.log(ct))
        elif re_type == 'none':
            a, alpha, = self.forward_process(p_type=re_type)
            beta = self.backward_process(p_type=re_type)
            log_likelihood = log(sum(alpha[-1, :]))

        like_value = self.likelihood
        finish_time = alpha.shape[0]
        log_likelihood_all = np.zeros(max_iternum + 1)
        log_likelihood_all[0] = log_likelihood
        count = 0
        increment_likelihood = 1
        while count < max_iternum and increment_likelihood > 1e-6:
            print("Iteration: ", count + 1)
            count = count + 1
            # Estimation for transition probability
            transition_p = np.zeros((self.Ns, self.Ns))
            for i in range(self.Ns):
                for j in range(self.Ns):
                    num = 0
                    den = 0
                    for t in range(finish_time - 1):
                        num = num + alpha[t, i] * self.Pt[i, j] * like_value[t + 1, j] * beta[t + 1, j]
                        den = den + alpha[t, i] * beta[t, i]
                    transition_p[i, j] = num / den
                transition_p[i, :] = transition_p[i, :] / sum(transition_p[i, :])
            # Estimation for initial probability
            initial_p = alpha[0, :] * beta[0, :]
            initial_p = initial_p / sum(initial_p)
            self.Pi = initial_p  # update initial probability
            self.Pt = transition_p  # update transition probability
            # re-compute forward and backward variables
            if re_type == 'posterior':
                a, alpha, c, ct = self.forward_process(p_type=re_type)
                beta = self.backward_process(p_type=re_type, scaling_factor=ct)
                log_likelihood_new = sum(np.log(ct))
            elif re_type == 'none':
                a, alpha, = self.forward_process(p_type=re_type)
                beta = self.backward_process(p_type=re_type)
                log_likelihood_new = np.log(sum(alpha[-1, :]))
            increment_likelihood = (log_likelihood_new - log_likelihood) / abs(log_likelihood)
            log_likelihood = log_likelihood_new
            log_likelihood_all[count] = log_likelihood_new

        log_likelihood_all = log_likelihood_all[0: count + 1]
        return transition_p, initial_p, log_likelihood_all


if __name__ == '__main__':
    import pickle

    state_num, anchor_num = 12, 8
    save_file = 'data1104.txt'
    # mean, cov, likelihood
    f = open(save_file, 'rb')
    trained_para = pickle.load(f)
    f.close()

    Pt = np.array([[.6, .4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [.25, .5, .25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, .2, .4, .2, 0, 0, .2, 0, 0, 0, 0, 0], [0, 0, .25, .5, .25, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, .25, .5, .25, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0.25, .5, 0, .25, 0, 0, 0, 0],
                   [0, 0, .1, 0, 0, 0, .4, .2, .15, 0, 0, 0.15], [0, 0, 0, 0, 0, 0.25, .25, .5, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, .25, 0, .5, .25, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, .4, .6, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .6, .4], [0, 0, 0, 0, 0, 0, .25, 0, 0, 0, .25, .5]])

    para_hmm = {'state_num': state_num, 'anchor_num': anchor_num, 'initial_p': 1 / state_num * np.ones(state_num),
                'transition_p': Pt, 'training_type': 'from_external',
                'mean': trained_para[0], 'cov': trained_para[1],
                'likelihood_from': 'from_external', 'likelihood': trained_para[2]}

    hmm = HiddenMarkov(para_hmm, specification=True)

    # parameter re-estimation
    pt_hmm, ip, max_likeli = hmm.para_estimation(max_iternum=2)

    print("The trained transition is:")
    print(pt_hmm)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='r')
    a1 = ax[0].matshow(Pt, cmap='viridis')
    a2 = ax[1].matshow(pt_hmm, cmap='viridis')
    plt.show()

