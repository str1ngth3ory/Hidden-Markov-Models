import unittest
import math

# from hmm_submission_solution import *
from hmm_submission import gaussian_prob, part_1_a, part_2_a, \
                           viterbi, multidimensional_viterbi

B_STATES = ['B1', 'B2', 'B3', 'Bend']
C_STATES = ['C1', 'C2', 'C3', 'Cend']
H_STATES = ['H1', 'H2', 'H3', 'Hend']

class HMMTestPart1(unittest.TestCase):
    """Test Part 1 of the HMM Submission
    """

    (b_prior_probs, b_transition_probs, b_emission_paras,
     c_prior_probs, c_transition_probs, c_emission_paras,
     h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()

    evidence_vector = [44, 51, 57, 63, 61, 60, 59]
    correct_sequence = {
        'BUY': ['B1', 'B2', 'B2', 'B2', 'B2', 'B2', 'B2'],
        'CAR': ['C1', 'C2', 'C2', 'C2', 'C2', 'C2', 'C2'],
        'HOUSE': ['H1', 'H2', 'H3', 'H3', 'H3', 'H3', 'H3']
    }
    

    part_1b_B = viterbi(evidence_vector,
                        B_STATES,
                        b_prior_probs,
                        b_transition_probs,
                        b_emission_paras)

    part_1b_C = viterbi(evidence_vector,
                        C_STATES,
                        c_prior_probs,
                        c_transition_probs,
                        c_emission_paras)

    part_1b_H = viterbi(evidence_vector,
                        H_STATES,
                        h_prior_probs,
                        h_transition_probs,
                        h_emission_paras)

    def test_3words_prior(self):
        b_prior = sum(self.b_prior_probs.values())
        c_prior = sum(self.c_prior_probs.values())
        h_prior = sum(self.h_prior_probs.values())
        total_prob = b_prior + c_prior + h_prior
        msg = ('incorrect prior probs. each word should be selected with '
               'equal probability. counted {}, should be 1').format(total_prob)
        self.assertAlmostEqual(1.0, total_prob, places=2, msg=msg)


    def test_buy_transition(self):
        for state, probs in self.b_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)


    def test_car_transition(self):
        for state, probs in self.c_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('CAR transition prob in state {} should sum to 1 '
                   '(get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)


    def test_house_transition(self):
        for state, probs in self.h_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('HOUSE transition prob in state {} should sum to 1 '
                   '(get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)


    def test_buy_emission(self):
        benchmark = {'B1': 0.1179, 'B2': 0.0001, 'B3': 0.0113}
        result = {s: gaussian_prob(40, self.b_emission_paras[s])
                    for s in benchmark.keys()}
        for state in benchmark:
            msg = ('incorrect mean and/or std for state {}. expect {} but '
                   'get {}').format(state, benchmark[state], result[state])
            self.assertAlmostEqual(benchmark[state], result[state], 3, msg)


    def test_car_emission(self):
        benchmark = {'C1': 0.0551, 'C2': 0.0254, 'C3': 0.0559}
        result = {s: gaussian_prob(40, self.c_emission_paras[s])
                    for s in benchmark.keys()}
        for state in benchmark:
            msg = ('incorrect mean and/or std for state {}. expect {} but '
                   'get {}').format(state, benchmark[state], result[state])
            self.assertAlmostEqual(benchmark[state], result[state], 3, msg)


    def test_house_emission(self):
        benchmark = {'H1': 0.0408, 'H2': 0.0405, 'H3': 7.3931e-7}
        result = {s: gaussian_prob(40, self.h_emission_paras[s])
                    for s in benchmark.keys()}
        for state in benchmark:
            msg = ('incorrect mean and/or std for state {}. expect {} but '
                   'get {}').format(state, benchmark[state], result[state])
            self.assertAlmostEqual(benchmark[state], result[state], 3, msg)


    def test_viterbi_probs(self):
        _, prob_b = self.part_1b_B
        _, prob_c = self.part_1b_C
        _, prob_h = self.part_1b_H
        msg = ('for evidence vector {}, the most likely'
               'word should be BUY').format(self.evidence_vector)
        self.assertEqual(max([prob_b, prob_c, prob_h]), prob_b, msg)
        msg = ('for evidence vector {}, the probability of BUY should be '
               '5.326e-13, get {}').format(self.evidence_vector, prob_b)
        self.assertAlmostEqual(5.326e-13, prob_b, 13, msg)


    def test_viterbi_sequence(self):
        seq_b, _ = self.part_1b_B
        seq_c, _ = self.part_1b_C
        seq_h, _ = self.part_1b_H
        msg = ('sequence for BUY is incorrect. expect {} but get {}'
               '').format(self.correct_sequence['BUY'], seq_b)
        self.assertEqual(self.correct_sequence['BUY'], seq_b, msg)
        msg = ('sequence for BUY is incorrect. expect {} but get {}'
               '').format(self.correct_sequence['BUY'], seq_b)
        self.assertEqual(self.correct_sequence['CAR'], seq_c, msg)
        msg = ('sequence for BUY is incorrect. expect {} but get {}'
               '').format(self.correct_sequence['BUY'], seq_b)
        self.assertEqual(self.correct_sequence['HOUSE'], seq_h, msg)


class HMMTestPart2(unittest.TestCase):
    """Test Part 2 of the HMM Submission
    """

    correct_emission = {
        'B1': (108.2, 17.314),
        'B2': (78.67, 1.886),
        'B3': (64.182, 5.573),
        'Bend': (None, None),
        'C1': (56.3, 10.659),
        'C2': (37.11, 4.306),
        'C3': (50., 7.826),
        'Cend': (None, None),
        'H1': (53.6, 7.392),
        'H2': (37.168, 8.875),
        'H3': (74.176, 8.347),
        'Hend': (None, None),
    }

    (b_prior_probs, b_transition_probs, b_emission_paras,
     c_prior_probs, c_transition_probs, c_emission_paras,
     h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()

    ev_right = [44, 51, 57, 63, 61, 60, 59]
    ev_left = [101, 95, 84, 77, 73, 68, 66]
    evidence_vector = list(zip(ev_right, ev_left))

    part_2b_B = multidimensional_viterbi(evidence_vector,
                                         B_STATES,
                                         b_prior_probs,
                                         b_transition_probs,
                                         b_emission_paras)

    part_2b_C = multidimensional_viterbi(evidence_vector,
                                         C_STATES,
                                         c_prior_probs,
                                         c_transition_probs,
                                         c_emission_paras)

    part_2b_H = multidimensional_viterbi(evidence_vector,
                                         H_STATES,
                                         h_prior_probs,
                                         h_transition_probs,
                                         h_emission_paras)

    def test_3words_prior(self):
        b_prior = sum(self.b_prior_probs.values())
        c_prior = sum(self.c_prior_probs.values())
        h_prior = sum(self.h_prior_probs.values())
        total_prob = b_prior + c_prior + h_prior
        msg = ('incorrect prior probs. each word should be selected with '
               'equal probability. counted {}, should be 1').format(total_prob)
        self.assertAlmostEqual(1.0, total_prob, places=2, msg=msg)


    def test_buy_transition(self):
        for state, probs in self.b_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=2, msg=msg)
            msg = ('left hand BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=2, msg=msg)


    def test_car_transition(self):
        for state, probs in self.c_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand CAR transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=2, msg=msg)
            msg = ('left hand CAR transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=2, msg=msg)


    def test_house_transition(self):
        for state, probs in self.h_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand HOUSE transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=2, msg=msg)
            msg = ('left hand HOUSE transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=2, msg=msg)


    def test_buy_emission(self):
        for state, paras in self.b_emission_paras.items():
            msg = ('left-hand emission probs for word BUY is incorrect. expect'
                  ' {}, get {}').format(self.correct_emission[state], paras[1])
            self.assertEqual(self.correct_emission[state], paras[1], msg)
    

    def test_car_emission(self):
        for state, paras in self.c_emission_paras.items():
            msg = ('left-hand emission probs for word CAR is incorrect. expect'
                  ' {}, get {}').format(self.correct_emission[state], paras[1])
            self.assertEqual(self.correct_emission[state], paras[1], msg)


    def test_house_emission(self):
        for state, paras in self.h_emission_paras.items():
            msg = ('left-hand emission probs for word HOUSE is incorrect. expect'
                  ' {}, get {}').format(self.correct_emission[state], paras[1])
            self.assertEqual(self.correct_emission[state], paras[1], msg)


    def test_viterbi_probs(self):
        _, prob_b = self.part_2b_B
        _, prob_c = self.part_2b_C
        _, prob_h = self.part_2b_H
        msg = ('for evidence vector {}, the most likely'
               'word should be BUY').format(self.evidence_vector)
        self.assertEqual(max([prob_b, prob_c, prob_h]), prob_b, msg)



if __name__ == "__main__":
    unittest.main(verbosity=1)
