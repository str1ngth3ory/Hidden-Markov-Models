
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: notebook.ipynb

import numpy as np
import operator

def part_1_a():
    """Provide probabilities for the word HMMs outlined below.
    Word BUY, CAR, and HOUSE.
    Review Udacity Lesson 8 - Video #29. HMM Training
    Returns:
        tuple() of
        (prior probabilities for all states for word BUY,
         transition probabilities between states for word BUY,
         emission parameters tuple(mean, std) for all states for word BUY,
         prior probabilities for all states for word CAR,
         transition probabilities between states for word CAR,
         emission parameters tuple(mean, std) for all states for word CAR,
         prior probabilities for all states for word HOUSE,
         transition probabilities between states for word HOUSE,
         emission parameters tuple(mean, std) for all states for word HOUSE,)
        Sample Format (not complete):
        (
            {'B1': prob_of_starting_in_B1, 'B2': prob_of_starting_in_B2, ...},
            {'B1': {'B1': prob_of_transition_from_B1_to_B1,
                    'B2': prob_of_transition_from_B1_to_B2,
                    'B3': prob_of_transition_from_B1_to_B3,
                    'Bend': prob_of_transition_from_B1_to_Bend},
             'B2': {...}, ...},
            {'B1': tuple(mean_of_B1, standard_deviation_of_B1),
             'B2': tuple(mean_of_B2, standard_deviation_of_B2), ...},
            {'C1': prob_of_starting_in_C1, 'C2': prob_of_starting_in_C2, ...},
            {'C1': {'C1': prob_of_transition_from_C1_to_C1,
                    'C2': prob_of_transition_from_C1_to_C2,
                    'C3': prob_of_transition_from_C1_to_C3,
                    'Cend': prob_of_transition_from_C1_to_Cend},
             'C2': {...}, ...}
            {'C1': tuple(mean_of_C1, standard_deviation_of_C1),
             'C2': tuple(mean_of_C2, standard_deviation_of_C2), ...}
            {'H1': prob_of_starting_in_H1, 'H2': prob_of_starting_in_H2, ...},
            {'H1': {'H1': prob_of_transition_from_H1_to_H1,
                    'H2': prob_of_transition_from_H1_to_H2,
                    'H3': prob_of_transition_from_H1_to_H3,
                    'Hend': prob_of_transition_from_H1_to_Hend},
             'H2': {...}, ...}
            {'H1': tuple(mean_of_H1, standard_deviation_of_H1),
             'H2': tuple(mean_of_H2, standard_deviation_of_H2), ...}
        )
    """

    # TODO: complete this function.

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    b_transition_probs = {
        'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0., 'Bend': 0.},
        'B2': {'B1': 0., 'B2': 0.625, 'B3': 0.375, 'Bend': 0.},
        'B3': {'B1': 0., 'B2': 0., 'B3': 0.625, 'Bend': 0.375},
        'Bend': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 1.},
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (41.750, 2.773),
        'B2': (58.625, 5.678),
        'B3': (53.125, 5.418),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0., 'Cend': 0.},
        'C2': {'C1': 0., 'C2': 0., 'C3': 1.0, 'Cend': 0.},
        'C3': {'C1': 0., 'C2': 0., 'C3': 0.800, 'Cend': 0.200},
        'Cend': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 1.},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (35.667, 4.899),
        'C2': (43.667, 1.700),
        'C3': (44.200, 7.341),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0., 'Hend': 0.},
        'H2': {'H1': 0., 'H2': 0.857, 'H3': 0.143, 'Hend': 0.},
        'H3': {'H1': 0., 'H2': 0., 'H3': 0.813, 'Hend': 0.187},
        'Hend': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 1.},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (45.333, 3.972),
        'H2': (34.952, 8.127),
        'H3': (67.438, 5.733),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous
        distribution should be zero, instead of the number outputted here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x.
        The probability of getting value "x" is zero because a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if list(para_tuple) == [None, None]:
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.
    Args:
        evidence_vector (list): List of right hand Y-axis positions (integer).
        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend', 'H1', 'H2', 'H3', 'Hend']
        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}
        transition_probs (dict): dictionary representing transitions from each
                                 state to every other valid state such as for the above
                                 states, there won't be a transition from 'B1' to 'H1'
        emission_paras (dict): parameters of Gaussian distribution
                                from each state.
    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )
    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.
    """
    if not evidence_vector:
        return (None, 0)

    T1 = np.zeros((len(states),len(evidence_vector)))
    T2 = np.zeros((len(states),len(evidence_vector)))

    for i in range(len(states)):

        T1[i,0] = prior_probs[states[i]]*gaussian_prob(evidence_vector[0],emission_paras[states[i]])

    for i in range(1, len(evidence_vector)):
        for j in range(len(states)):
            temp = 0
            max_idx = -1
            state_range = range(len(states))
            for k in state_range:
                if states[j] in transition_probs[states[k]]:
                    temp = T1[k,i-1]*transition_probs[states[k]][states[j]]*gaussian_prob(evidence_vector[i],emission_paras[states[j]])
                else:
                    temp = 0

                if temp > T1[j,i]:
                    T1[j,i] = temp
                    T2[j,i] = k

    sequence = []
    probability = np.max(T1[:,-1])

    curr_idx = np.argmax(T1[:,-1],axis=0)

    for i in range(len(evidence_vector),0,-1):
        sequence.append(states[curr_idx])
        if i > 1:
            curr_idx = int(T2[curr_idx,i-1])
    sequence.reverse()

    if probability == 0:
        return (None, 0)
    else:
        return sequence, probability

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################


def part_2_a():
    """Provide probabilities for the word HMMs outlined below.
    Now, at each time frame you are given 2 observations (right hand Y
    position & left hand Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for left hand, create
    a tuple of (right-y, left-y) to represent high-dimension transition &
    emission probabilities.
    """

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0.625, 0.700), 'B2': (0.375, 0.300), 'B3': (0., 0.), 'Bend': (0., 0.)},
        'B2': {'B1': (0., 0.), 'B2': (0.625, 0.050), 'B3': (0.375, 0.950), 'Bend': (0., 0.)},
        'B3': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0.625, 0.727), 'Bend': (0.125, 0.091), 'C1': (0.125,0.091), 'H1': (0.125,0.091)},
        'Bend': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (1., 1.)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(41.750, 2.773), (108.200, 17.314)],
        'B2': [(58.625, 5.678), (78.670, 1.886)],
        'B3': [(53.125, 5.418), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': (0.667, 0.700), 'C2': (0.333, 0.300), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'C2': {'C1': (0., 0.), 'C2': (0., 0.625), 'C3': (1.000, 0.375), 'Cend': (0., 0.)},
        'C3': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0.800, 0.625), 'Cend': (0.067, 0.125), 'B1': (0.067,0.125), 'H1': (0.067,0.125)},
        'Cend': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (1., 1.)},
    }
    c_emission_paras = {
        'C1': [(35.667, 4.899), (56.300, 10.659)],
        'C2': [(43.667, 1.700), (37.110, 4.306)],
        'C3': [(44.200, 7.341), (50.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    h_transition_probs = {
        'H1': {'H1': (0.667, 0.700), 'H2': (0.333, 0.300), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'H2': {'H1': (0., 0.), 'H2': (0.857, 0.842), 'H3': (0.143, 0.158), 'Hend': (0., 0.)},
        'H3': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0.813, 0.824), 'Hend': (0.062, 0.059), 'B1': (0.062, 0.059), 'C1': (0.062, 0.059)},
        'Hend': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (1., 1.)},
    }
    h_emission_paras = {
        'H1': [(45.333, 3.972), (53.600, 7.392)],
        'H2': [(34.952, 8.127), (37.168, 8.875)],
        'H3': [(67.438, 5.733), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.
    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    Evidence vector is a list of tuples where the first element of each tuple is the right
    hand coordinate and the second element is the left hand coordinate.
    """
    if not evidence_vector:
        return (None, 0)

    T1 = np.zeros((len(states),len(evidence_vector)))
    T2 = np.zeros((len(states),len(evidence_vector)))

    for i in range(len(states)):

        T1[i,0] = prior_probs[states[i]]*gaussian_prob(evidence_vector[0][0],emission_paras[states[i]][0])*gaussian_prob(evidence_vector[0][1],emission_paras[states[i]][1])

    for i in range(1, len(evidence_vector)):
        for j in range(len(states)):
            temp = 0
            max_idx = -1
            state_range = range(len(states))
            for k in state_range:
                if states[j] in transition_probs[states[k]]:
                    temp = T1[k,i-1]*transition_probs[states[k]][states[j]][0]*transition_probs[states[k]][states[j]][1]*gaussian_prob(evidence_vector[i][0],emission_paras[states[j]][0])*gaussian_prob(evidence_vector[i][1],emission_paras[states[j]][1])
                else:
                    temp = 0

                if temp > T1[j,i]:
                    T1[j,i] = temp
                    T2[j,i] = k

    sequence = []
    probability = np.max(T1[:,-1])

    curr_idx = np.argmax(T1[:,-1],axis=0)

    for i in range(len(evidence_vector),0,-1):
        sequence.append(states[curr_idx])
        if i > 1:
            curr_idx = int(T2[curr_idx,i-1])
    sequence.reverse()
    print(sequence, probability)
    if probability == 0:
        return (None, 0)
    else:
        return sequence, probability

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def return_your_name():
    """Return your name
    """
    # TODO: finish this
    return "Zi Sang"