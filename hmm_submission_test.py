import unittest

# from hmm_submission import part_1_a, part_2_a
# from hmm_submission import viterbi, multidimensional_viterbi
from hmm_submission_solution import *

class TestPart1a(unittest.TestCase):

    (b_prior_probs, b_transition_probs, b_emission_paras,
    c_prior_probs, c_transition_probs, c_emission_paras,
    h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()


    def test_prior(self):
        b_prior = sum(self.b_prior_probs.values())
        c_prior = sum(self.c_prior_probs.values())
        h_prior = sum(self.h_prior_probs.values())
        total_prob = b_prior + c_prior + h_prior
        msg = ('incorrect prior probs. each word should be selected with '
               'equal probability. counted {}, should be 1').format(total_prob)
        self.assertAlmostEqual(1.0, total_prob, places=2, msg=msg)


    def test_b_emision(self):
        mean, std = self.b_emission_paras['B1']
        msg = 'incorrect mean for letter BUY, state B1'
        self.assertEqual(8143732372720334548, hash(str(mean)), msg)
        msg = 'incorrect std for letter BUY, state B1'
        self.assertEqual(4909260699532052808, hash(str(std)), msg)

        mean, std = self.b_emission_paras['B2']
        msg = 'incorrect mean for letter BUY, state B2'
        self.assertEqual(8053884863761162022, hash(str(mean)), msg)
        msg = 'incorrect std for letter BUY, state B2'
        self.assertEqual(-8409099901164351883, hash(str(std)), msg)

        mean, std = self.b_emission_paras['B3']
        msg = 'incorrect mean for letter BUY, state B3'
        self.assertEqual(-5500469697994766098, hash(str(mean)), msg)
        msg = 'incorrect std for letter BUY, state B3'
        self.assertEqual(-8409097901178351951, hash(str(std)), msg)


    def test_c_emision(self):
        mean, std = self.c_emission_paras['C1']
        msg = 'incorrect mean for letter CAR, state C1'
        self.assertEqual(1761083565526287931, hash(str(mean)), msg)
        msg = 'incorrect std for letter CAR, state C1'
        self.assertEqual(-5302976700826216731, hash(str(std)), msg)

        mean, std = self.c_emission_paras['C2']
        msg = 'incorrect mean for letter CAR, state C2'
        self.assertEqual(-3528458061032876368, hash(str(mean)), msg)
        msg = 'incorrect std for letter CAR, state C2'
        self.assertEqual(163512108432620419, hash(str(std)), msg)

        mean, std = self.c_emission_paras['C3']
        msg = 'incorrect mean for letter CAR, state C3'
        self.assertEqual(-5323582098098394060, hash(str(mean)), msg)
        msg = 'incorrect std for letter CAR, state C3'
        self.assertEqual(4718540918289031730, hash(str(std)), msg)


    def test_h_emision(self):
        mean, std = self.h_emission_paras['H1']
        msg = 'incorrect mean for letter HOUSE, state H1'
        self.assertEqual(22004294861154160, hash(str(mean)), msg)
        msg = 'incorrect std for letter HOUSE, state H1'
        self.assertEqual(1803143499295917918, hash(str(std)), msg)

        mean, std = self.h_emission_paras['H2']
        msg = 'incorrect mean for letter HOUSE, state H2'
        self.assertEqual(-6681881302971367059, hash(str(mean)), msg)
        msg = 'incorrect std for letter HOUSE, state H2'
        self.assertEqual(-2387580281827102941, hash(str(std)), msg)

        mean, std = self.h_emission_paras['H3']
        msg = 'incorrect mean for letter HOUSE, state H3'
        self.assertEqual(-168079895755008500, hash(str(mean)), msg)
        msg = 'incorrect std for letter HOUSE, state H3'
        self.assertEqual(-8409100901160351893, hash(str(std)), msg)


    def test_b_transition(self):
        for state, probs in self.b_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)


    def test_c_transition(self):
        for state, probs in self.c_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('CAR transition prob in state {} should sum to 1 '
                   '(get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)


    def test_h_transition(self):
        for state, probs in self.h_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('HOUSE transition prob in state {} should sum to 1 '
                   '(get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)


class TestPart1b(unittest.TestCase):

    b_states = ['B1', 'B2', 'B3', 'Bend']
    c_states = ['C1', 'C2', 'C3', 'Cend']
    h_states = ['H1', 'H2', 'H3', 'Hend']

    (b_prior_probs, b_transition_probs, b_emission_paras,
    c_prior_probs, c_transition_probs, c_emission_paras,
    h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()

    states = b_states + c_states + h_states
    prior = b_prior_probs
    prior.update(c_prior_probs)
    prior.update(h_prior_probs)

    trans = b_transition_probs
    trans.update(c_transition_probs)
    trans.update(h_transition_probs)

    emiss = b_emission_paras
    emiss.update(c_emission_paras)
    emiss.update(h_emission_paras)

    def test_viterbi_case1(self):
        evidence = []
        seq, prob = viterbi(evidence,
                            self.states,
                            self.prior,
                            self.trans,
                            self.emiss)
        msg = ('when evidence is an empty list, return "None" or [], '
                'get {}').format(seq)
        self.assertTrue(seq in [None, []], msg)
        msg = ('when evidence is an empty list, return prob=0.0, '
                'get {}').format(prob)
        self.assertTrue(prob == 0., msg)


    def test_viterbi_case2(self):
        evidence = [50]
        prob_ans = 0.0167710014759
        seq_ans = ['H1']

        seq, prob = viterbi(evidence,
                            self.states,
                            self.prior,
                            self.trans,
                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=7)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_case3(self):
        evidence = [40]
        prob_ans = 0.0392573919648
        seq_ans = ['B1']

        seq, prob = viterbi(evidence,
                            self.states,
                            self.prior,
                            self.trans,
                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=7)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_realsample1(self):
        """
        Extracted from BOSTON ASL dataset: video 51, frame 17-23
        Actual words: BUY
        """
        evidence = [44, 51, 57, 63, 61, 60, 59]
        prob_ans = 3.81226460569e-11
        seq_ans = ['B1', 'B2', 'B2', 'B2', 'B2', 'B2', 'B2']
        seq, prob = viterbi(evidence,
                            self.states,
                            self.prior,
                            self.trans,
                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=14)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_realsample2(self):
        """
        Extracted from BOSTON ASL dataset: video 48, frame 27-35
        Actual words: CAR
        """
        evidence = [45, 45, 46, 48, 51, 51, 49, 45, 42]
        prob_ans = 6.3377763423e-13
        seq_ans = ['H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1']
        seq, prob = viterbi(evidence,
                            self.states,
                            self.prior,
                            self.trans,
                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=17)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_realsample3(self):
        """
        Extracted from BOSTON ASL dataset: video 47, frame 21-30
        Actual words: CAR
        """
        evidence = [47, 39, 32, 34, 36, 42, 42, 42, 34, 25]
        prob_ans = 6.07498784245e-16
        seq_ans = ['H1', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2']
        seq, prob = viterbi(evidence,
                            self.states,
                            self.prior,
                            self.trans,
                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=20)
        self.assertEqual(seq_ans, seq)


class TestPart2a(unittest.TestCase):

    (b_prior_probs, b_transition_probs, b_emission_paras,
     c_prior_probs, c_transition_probs, c_emission_paras,
     h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()


    def test_prior(self):
        b_prior = sum(self.b_prior_probs.values())
        c_prior = sum(self.c_prior_probs.values())
        h_prior = sum(self.h_prior_probs.values())
        total_prob = b_prior + c_prior + h_prior
        msg = ('incorrect prior probs. each word should be selected with '
               'equal probability. counted {}, should be 1').format(total_prob)
        self.assertAlmostEqual(1.0, total_prob, places=2, msg=msg)


    def test_b_emision(self):
        right, left = self.b_emission_paras['B1']
        rmean, rstd = right
        msg = 'incorrect mean for letter BUY, state B1, right hand'
        self.assertEqual(8143732372720334548, hash(str(rmean)), msg)
        msg = 'incorrect std for letter BUY, state B1, right hand'
        self.assertEqual(4909260699532052808, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter BUY, state B1, left hand'
        self.assertEqual(4461908972760738592, hash(str(lmean)), msg)
        msg = 'incorrect std for letter BUY, state B1, left hand'
        self.assertEqual(1458919922068683000, hash(str(lstd)), msg)


        right, left = self.b_emission_paras['B2']
        rmean, rstd = right
        msg = 'incorrect mean for letter BUY, state B2, right hand'
        self.assertEqual(8053877863755162099, hash(str(rmean)), msg)
        msg = 'incorrect std for letter BUY, state B2, right hand'
        self.assertEqual(-5302976700817216765, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter BUY, state B2, left hand'
        self.assertEqual(4272007845397481125, hash(str(lmean)), msg)
        msg = 'incorrect std for letter BUY, state B2, left hand'
        self.assertEqual(-6431361174054364148, hash(str(lstd)), msg)


        right, left = self.b_emission_paras['B3']
        rmean, rstd = right
        msg = 'incorrect mean for letter BUY, state B3, right hand'
        self.assertEqual(-5323585098128394121, hash(str(rmean)), msg)
        msg = 'incorrect std for letter BUY, state B3, right hand'
        self.assertEqual(-5302988700954216970, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter BUY, state B3, left hand'
        self.assertEqual(-8843841129772418303, hash(str(lmean)), msg)
        msg = 'incorrect std for letter BUY, state B3, left hand'
        self.assertEqual(-8409098901180351943, hash(str(lstd)), msg)


    def test_c_emision(self):
        right, left = self.c_emission_paras['C1']
        rmean, rstd = right
        msg = 'incorrect mean for letter CAR, state C1, right hand'
        self.assertEqual(1761083565526287931, hash(str(rmean)), msg)
        msg = 'incorrect std for letter CAR, state C1, right hand'
        self.assertEqual(-5302976700826216731, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter CAR, state C1, left hand'
        self.assertEqual(-1169019967607120632, hash(str(lmean)), msg)
        msg = 'incorrect std for letter CAR, state C1, left hand'
        self.assertEqual(-4544859283256213909, hash(str(lstd)), msg)


        right, left = self.c_emission_paras['C2']
        rmean, rstd = right
        msg = 'incorrect mean for letter CAR, state C2, right hand'
        self.assertEqual(-5323575098000393896, hash(str(rmean)), msg)
        msg = 'incorrect std for letter CAR, state C2, right hand'
        self.assertEqual(-6431361174046364063, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter CAR, state C2, left hand'
        self.assertEqual(-7196902500767082077, hash(str(lmean)), msg)
        msg = 'incorrect std for letter CAR, state C2, left hand'
        self.assertEqual(-5302985700937216978, hash(str(lstd)), msg)


        right, left = self.c_emission_paras['C3']
        rmean, rstd = right
        msg = 'incorrect mean for letter CAR, state C3, right hand'
        self.assertEqual(8465069163979809992, hash(str(rmean)), msg)
        msg = 'incorrect std for letter CAR, state C3, right hand'
        self.assertEqual(-2387580281828102926, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter CAR, state C3, left hand'
        self.assertEqual(-1169013967511120355, hash(str(lmean)), msg)
        msg = 'incorrect std for letter CAR, state C3, left hand'
        self.assertEqual(4718547918429032024, hash(str(lstd)), msg)


    def test_h_emision(self):
        right, left = self.h_emission_paras['H1']
        rmean, rstd = right
        msg = 'incorrect mean for letter HOUSE, state H1, right hand'
        self.assertEqual(22004294861154160, hash(str(rmean)), msg)
        msg = 'incorrect std for letter HOUSE, state H1, right hand'
        self.assertEqual(1803143499295917918, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter HOUSE, state H1, left hand'
        self.assertEqual(-1169016967513120372, hash(str(lmean)), msg)
        msg = 'incorrect std for letter HOUSE, state H1, left hand'
        self.assertEqual(4718540918302031706, hash(str(lstd)), msg)


        right, left = self.h_emission_paras['H2']
        rmean, rstd = right
        msg = 'incorrect mean for letter HOUSE, state H2, right hand'
        self.assertEqual(-6681881302971367059, hash(str(rmean)), msg)
        msg = 'incorrect std for letter HOUSE, state H2, right hand'
        self.assertEqual(-2387580281827102941, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter HOUSE, state H2, left hand'
        self.assertEqual(-678093097655470307, hash(str(lmean)), msg)
        msg = 'incorrect std for letter HOUSE, state H2, left hand'
        self.assertEqual(-2387573281824102997, hash(str(lstd)), msg)


        right, left = self.h_emission_paras['H3']
        rmean, rstd = right
        msg = 'incorrect mean for letter HOUSE, state H3, right hand'
        self.assertEqual(-168079895755008500, hash(str(rmean)), msg)
        msg = 'incorrect std for letter HOUSE, state H3, right hand'
        self.assertEqual(-8409100901160351893, hash(str(rstd)), msg)

        lmean, lstd = left
        msg = 'incorrect mean for letter HOUSE, state H3, left hand'
        self.assertEqual(-8143741737273793857, hash(str(lmean)), msg)
        msg = 'incorrect std for letter HOUSE, state H3, left hand'
        self.assertEqual(-2387582281837102921, hash(str(lstd)), msg)


    def test_b_transition(self):
        for state, probs in self.b_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=3, msg=msg)
            msg = ('left hand BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=3, msg=msg)


    def test_c_transition(self):
        for state, probs in self.c_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand CAR transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=2, msg=msg)
            msg = ('left hand CAR transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=3, msg=msg)


    def test_h_transition(self):
        for state, probs in self.h_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand HOUSE transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=3, msg=msg)
            msg = ('left hand HOUSE transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=2, msg=msg)


class TestPart2b(unittest.TestCase):

    b_states = ['B1', 'B2', 'B3', 'Bend']
    c_states = ['C1', 'C2', 'C3', 'Cend']
    h_states = ['H1', 'H2', 'H3', 'Hend']

    (b_prior_probs, b_transition_probs, b_emission_paras,
    c_prior_probs, c_transition_probs, c_emission_paras,
    h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()

    states = b_states + c_states + h_states
    prior = b_prior_probs
    prior.update(c_prior_probs)
    prior.update(h_prior_probs)

    trans = b_transition_probs
    trans.update(c_transition_probs)
    trans.update(h_transition_probs)

    emiss = b_emission_paras
    emiss.update(c_emission_paras)
    emiss.update(h_emission_paras)

    def test_viterbi_case1(self):
        evidence = []
        seq, prob = viterbi(evidence,
                            self.states,
                            self.b_prior_probs,
                            self.b_transition_probs,
                            self.b_emission_paras)
        msg = ('when evidence is an empty list, return "None" or [], '
                'get {}').format(seq)
        self.assertTrue(seq in [None, []], msg)
        msg = ('when evidence is an empty list, return prob=0.0, '
                'get {}').format(prob)
        self.assertTrue(prob == 0., msg)


    def test_viterbi_case2(self):
        evidence = [(50, 100)]
        prob_ans = 1.18078362571e-05
        seq_ans = ['B1']

        seq, prob = multidimensional_viterbi(evidence,
                                            self.states,
                                            self.prior,
                                            self.trans,
                                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=9)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_case3(self):
        evidence = [(40, 40)]
        prob_ans = 0.000213189518752
        seq_ans = ['C1']

        seq, prob = multidimensional_viterbi(evidence,
                                            self.states,
                                            self.prior,
                                            self.trans,
                                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=15)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_realsample1(self):
        """
        Extracted from BOSTON ASL dataset: video 51, frame 17-23
        Actual words: BUY
        """
        right_y = [44, 51, 57, 63, 61, 60, 59]
        left_y = [101, 95, 84, 77, 73, 68, 66]
        evidence = list(zip(right_y, left_y))

        prob_ans = 1.11584340542e-30
        seq_ans = ['B1', 'B1', 'B2', 'B2', 'B3', 'B3', 'B3']
        seq, prob = multidimensional_viterbi(evidence,
                                            self.states,
                                            self.prior,
                                            self.trans,
                                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=34)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_realsample2(self):
        """
        Extracted from BOSTON ASL dataset: video 48, frame 27-35
        Actual words: CAR
        """
        right_y = [45, 45, 46, 48, 51, 51, 49, 45, 42]
        left_y = [47, 45, 43, 45, 48, 48, 54, 61, 67]
        evidence = list(zip(right_y, left_y))

        prob_ans = 5.80105906918e-28
        seq_ans = ['H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1']
        seq, prob = multidimensional_viterbi(evidence,
                                            self.states,
                                            self.prior,
                                            self.trans,
                                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=32)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_realsample3(self):
        """
        Extracted from BOSTON ASL dataset: video 47, frame 21-30
        Actual words: CAR
        """
        right_y = [47, 39, 32, 34, 36, 42, 42, 42, 34, 25]
        left_y = [67, 62, 56, 48, 43, 43, 44, 46, 52, 65]
        evidence = list(zip(right_y, left_y))

        prob_ans = 1.17739941205e-34
        seq_ans = ['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1']
        seq, prob = multidimensional_viterbi(evidence,
                                            self.states,
                                            self.prior,
                                            self.trans,
                                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=38)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_phrase1(self):
        """
        Extracted from BOSTON ASL dataset: video 45, frame 36-56
        Actual words: BUY HOUSE
        """
        right_y = [48, 53, 54, 53, 48, 48, 48, 53, 48, 48, 46, 46,
                    53, 58, 66, 72, 71, 76, 76, 76, 76]
        left_y = [85, 74, 70, 66, 64, 64, 64, 60, 60, 60, 50, 50,
                    57, 67, 74, 85, 84, 89, 93, 93, 94]
        evidence = list(zip(right_y, left_y))

        prob_ans = 1.60813252848e-76
        seq_ans = ['B1', 'B2', 'B3', 'B3', 'B3', 'B3', 'B3',
                    'B3', 'H1', 'H1','H1', 'H1', 'H2', 'H3',
                    'H3', 'H3', 'H3', 'H3', 'H3', 'H3', 'H3']
        seq, prob = multidimensional_viterbi(evidence,
                                            self.states,
                                            self.prior,
                                            self.trans,
                                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=80)
        self.assertEqual(seq_ans, seq)


    def test_viterbi_phrase2(self):
        """
        Extracted from BOSTON ASL dataset: video 47, frame 12-30
        Actual words: BUY CAR
        """
        right_y = [42, 40, 41, 43, 52, 55, 59, 60, 55, 47, 39, 32,
                    34, 36, 42, 42, 42, 34, 25]
        left_y = [138, 133, 123, 115, 104, 91, 76, 70, 67, 67, 62,
                    56, 48, 43, 43, 44, 46, 52, 65]
        evidence = list(zip(right_y, left_y))

        prob_ans = 2.73935869674e-72
        seq_ans = ['B1', 'B1', 'B1', 'B1', 'B1', 'B1', 'B2',
                    'B3','B3', 'B3', 'C1', 'C1', 'C1', 'C1',
                    'C1', 'C1', 'C1', 'C1', 'C1']
        seq, prob = multidimensional_viterbi(evidence,
                                            self.states,
                                            self.prior,
                                            self.trans,
                                            self.emiss)
        self.assertAlmostEqual(prob_ans, prob, places=76)
        self.assertEqual(seq_ans, seq)


if __name__ == "__main__":
    unittest.main(verbosity=2)
