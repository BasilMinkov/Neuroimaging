import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)


def forward_backward(observations, states, start_probability, transition_probability, emission_probability, end_state):

  # vectors
  forward = [] # a vector to keep forward members
  forward_save = {} # a dict to keep forward members for previous step
  backward = [] # a vector to keep backward members
  backward_save = {} # a dict to keep backward members for previous step

  # make a product for a forward procedure
  for i, observation in enumerate(observations):

    forward_current = {} # a dict to keep forward probability members for i-th observation

    for state in states:
        if i == 0: # initial state term
            forward_current[state] = start_probability[state] * emission_probability[state][observation]
        else: # following state terms
            # transition by emission multiplication
            forward_current[state] = sum(forward_save[j] * transition_probability[j][state] for j in states) * emission_probability[state][observation]

    forward.append(forward_current) # append current term to product
    forward_save = forward_current # save i-th member for i+1th step production

  # make a product for a backward procedure
  for i, observation in enumerate([0] + observations[:0:-1]):

    backward_current = {} # a dict to keep backward probability members for i-th observation

    for state in states: 
        if i == 0: # initial state term
            backward_current[state] = transition_probability[state][end_state]
        else: # following state terms
            # transition by emission multiplication
            backward_current[state] = sum(transition_probability[state][j] * emission_probability[j][observation] * backward_save[j] for j in states)

    backward.append(backward_current) # append current term to product
    backward_save = backward_current # save i-th member i+1th step production

  backward = backward[::-1] # reverse backward product

  # calculate forward and backward probabilities from polynomials
  p_forward = sum(forward_current[j] * transition_probability[j][end_state] for j in states)
  p_backward = sum(start_probability[j] * emission_probability[j][observations[0]] * backward_current[j] for j in states)

  # calculate posterior probability based on forward/backward polynomials and normalize it by forward probability
  p_posterior = {}
  for i, observation in enumerate(observations):
    p_posterior[observation] = {state: forward[i][state] * backward[i][state] / p_forward for state in states}

  return p_posterior


def plot_posterior(seq, p_posterior):

  # for i, state in enumerate(states):
  #   p_state = [p_posterior[symbol][state] for symbol in seq['Symbol']]
  #   plt.subplot(2, 1, i+1)
  #   plt.plot(p_state, "-")
  #   plt.title(state)
  #   plt.ylabel("Probability")
  # plt.xlabel("#Sympol")
  # plt.show()

  for i, state in enumerate(states):
    p_state = []
    [p_posterior[symbol][state] for j, symbol enumerate(seq['Symbol'])]
  #   plt.subplot(2, 1, i+1)
  #   plt.plot(p_state, "-")
  #   plt.title(state)
  #   plt.ylabel("Probability")
  #   plt.xlabel("#Sympol")
  # plt.show()


def plot_true_and_predicted(seq, p_posterior):

  state_true = []
  state_pred = []

  for state in seq['State']:
    if state == "St1":
      state_true.append(1)
    else:
      state_true.append(2)

  for symbol in seq['Symbol']:
    if p_posterior[symbol]["St1"] > p_posterior[symbol]["St2"]:
      state_pred.append(1)
    else:
      state_pred.append(2)

  state_true = np.array(state_true)
  state_pred = np.array(state_pred)

  a = state_true == 1
  b = state_pred == 1
  Tp = sum(a == b)

  a = state_true == 2
  b = state_pred == 2
  Tn = sum(a == b)

  a = state_true == 1
  b = state_pred == 2
  Fp = sum(a == b)

  a = state_true == 2
  b = state_pred == 1
  Fn = sum(a == b)

  Sp = Tn / (Tn+Fp)
  Sn = Tp / (Tp+Fn)

  plt.subplot(2, 1, 1)
  plt.plot(state_true)
  plt.title("True Seq")
  plt.ylabel("State")
  plt.subplot(2, 1, 2)
  plt.plot(state_pred)
  plt.title(f"Predicted Seq (Sp = {Sp}; Sn = {Sn})")
  plt.ylabel("State")
  plt.xlabel("#Symbol")
  plt.show()


if __name__ == "__main__":


  # states = ('Healthy', 'Fever')
  # end_state = 'E'
   
  # observations = ('normal', 'cold', 'dizzy')
   
  # start_probability = {'Healthy': 0.6, 'Fever': 0.4}
   
  # transition_probability = {
  #    'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
  #    'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
  #    }
   
  # emission_probability = {
  #    'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
  #    'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
  #    }

  seq = pd.read_csv("seq.csv")
 
  observations = ["a", "b", "c"]
  end_state = "E"

  states = ["St1", "St2"]

  transition_probability = {
    "St1" : {"St1" : 0.981, "St2" : 0.019, "E" : 0.001}, 
    "St2" : {"St1" : 0.007, "St2" : 0.992, "E" : 0.001},
  }

  emission_probability = {
    "St1" : {"a" : 0.076, "b" : 0.802, "c" : 0.122}, 
    "St2" : {"a" : 0.179, "b" : 0.151, "c" : 0.669},
  }

  start_probability = {"St1" : 0.723, "St2" : 0.277}

  p_posterior = forward_backward(observations, states, start_probability, transition_probability, emission_probability,end_state)
  print(p_posterior)
  plot_posterior(seq, p_posterior)
  plot_true_and_predicted(seq, p_posterior)