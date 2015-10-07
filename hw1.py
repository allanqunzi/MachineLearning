import time
import logger
import numpy as np
import scipy.sparse as sps

logger = logger.createLogger("hw1_q3")

class multivariateNaiveBayes(object):
    """
    """
    def __init__(self, data_folder, groups = 20, vocabulary = 61188):
        self.folder = data_folder
        self.groups = groups
        self.vocabulary = vocabulary
        self.trained = False
        self.log_pi_c = np.zeros(self.groups+1, dtype = np.float64)
        self.N_c = np.ones(self.groups+1, dtype = np.uint16) # using np.ones means add-one smoothing
        self.N_jc = np.ones((self.groups+1, self.vocabulary+1), dtype=np.uint16) # using np.ones means add-one smoothing
        self.log_theta_jc = np.zeros((self.groups+1, self.vocabulary+1), dtype=np.complex_)# real:log(theta);imag:log(1-theta)
        self.sps_real = None
        self.sps_imag = None

    def train(self, data_filename, label_filename, docs = 11269):
        dpath = self.folder + data_filename
        lpath = self.folder + label_filename
        train_label = np.loadtxt(lpath, dtype = np.uint8, usecols = (0,))
        train_label = np.concatenate((np.array([0]), train_label)) # add one element in front
        for item in train_label:
            self.N_c[item] += 1
        temp = 1.0/np.float64(docs + self.groups)
        self.log_pi_c = np.log(self.N_c * temp)

        # calculating N_jc
        train_data = np.loadtxt(dpath, dtype = np.uint16, usecols = (0,1))
        for row in train_data:
            self.N_jc[train_label[row[0]]][row[1]] += 1

        # calculating log_theta_jc
        for i in range(1, self.groups+1, 1):
            temp = 1.0/np.float64(self.N_c[i] + 1) # denominator = N_c + 2
            self.log_theta_jc[i].real = self.N_jc[i] * temp # theta_jc = N_jc / (N_c + 2)
            self.log_theta_jc[i].imag = np.log(1.0 - self.log_theta_jc[i].real)
            self.log_theta_jc[i].real = np.log(self.log_theta_jc[i].real)

        self.trained = True
        logger.info("Training is finished.")
        self.sps_real = sps.csr_matrix(self.log_theta_jc.real)
        self.sps_imag = sps.csr_matrix(self.log_theta_jc.imag)

    def test(self, data_filename, docs = 7505):
        if not self.trained:
            logger.warn("This instance has not been trained yet.")
        dpath = self.folder + data_filename
        doc_vecs = self._doc_vector_generator(dpath)
        test_prob = np.zeros((docs+1, self.groups+1), dtype = np.float64)
        prediction = np.zeros(docs+1, dtype = np.uint8)

        for i in range(1, docs+1, 1): # for each doc in the test docs
            if i%100 == 0:
                print "i = ", i
            doc_vec1, doc_vec2 = doc_vecs.next() # prepare the ith doc bool vector (length = vocabulary + 1)
            for j in range(1, self.groups+1, 1): # for each group
                test_prob[i][j] = self.log_pi_c[j] # probability in this group
                test_prob[i][j] += doc_vec1.dot(self.sps_real[j].transpose())[0,0]
                test_prob[i][j] += doc_vec2.dot(self.sps_imag[j].transpose())[0,0]
            prediction[i] = test_prob[i][1:].argmax(axis = 0) + 1       # assign the index which has greatest prob

        return test_prob, prediction

    def score(self, label_filename, predict):
        lpath = self.folder + label_filename
        test_label = np.loadtxt(lpath, dtype = np.uint8, usecols = (0,))
        test_label = np.concatenate((np.array([0]), test_label)) # add one element in front
        assert (test_label.shape[0] == predict.shape[0]), "The length of the array test_label and that of predict is not the same."

        correct_positives = np.zeros(self.groups+1, dtype = np.uint16)
        all_positives = np.zeros(self.groups+1, dtype = np.uint16)
        should_be_positives = np.zeros(self.groups+1, dtype = np.uint16)
        all_positives[0] = 1 # avoid dividing by zero
        should_be_positives[0] = 1 # avoid dividing by zero

        for i in range(1, test_label.shape[0], 1):
            should_be_positives[test_label[i]] += 1
            all_positives[predict[i]] += 1
            if test_label[i] == predict[i]:
                correct_positives[test_label[i]] += 1

        precision = np.float64(correct_positives)/np.float64(all_positives)
        recall = np.float64(correct_positives)/np.float64(should_be_positives)
        precision[0] = 1.0 # avoid dividing by zero
        score = (2.0 * precision * recall)/(precision + recall)
        logger.info("F_1 score for rec.sport.hockey 11 = %f", score[11])
        logger.info("The macro-averaged score = %f", score.sum()/np.float64(self.groups) )
        return precision, recall, score

    def _doc_vector_generator(self, path):
        data = np.loadtxt(path, dtype = np.uint16, usecols = (0,1))
        length = len(data[:,0])
        i = 0
        cur_doc = 0
        while i != length:
            #doc_vector = np.zeros(self.vocabulary+1, dtype = np.bool_)
            doc_vector = np.zeros(self.vocabulary+1, dtype = np.int8)

            cur_doc += 1
            while i < length and data[i][0] == cur_doc:
                doc_vector[data[i][1]] = 1 #True
                i += 1
            yield sps.csr_matrix(doc_vector), sps.csr_matrix(1 - doc_vector)


if __name__ == '__main__':

    folder = './20news-bydate/matlab/' # don't forget the last slash

    bayes_classifier = multivariateNaiveBayes(folder)

    train_time = time.clock()
    bayes_classifier.train('train.data','train.label')
    train_time = time.clock() - train_time
    logger.info('time used for training: %f', train_time)

    test_time = time.clock()
    test_res, pred_res = bayes_classifier.test('test.data')
    test_time = time.clock() - test_time
    logger.info('time used for testing: %f', test_time)

    np.savetxt('test_res.txt', test_res)
    logger.info('The log probabilities for the docs in the test set are saved as test_res.txt')

    np.savetxt('pred_res.txt', pred_res)
    logger.info('The predicted group for each doc is saved as pred_res.txt')

    score_time = time.clock()
    p, r, s = bayes_classifier.score('test.label', pred_res)
    score_time = time.clock() - score_time
    logger.info('time used for calculating score: %f', score_time)

    np.savetxt('precision.txt', p)
    np.savetxt('recall.txt', r)
    np.savetxt('score.txt', s)
    logger.info('precision, recall, score are saved as precision.txt, recall.txt, score.txt, respectively.')













