import time
import logger
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import grid_search
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm

logger = logger.createLogger("hw2_q3")

class FeatureGenerator(object):
    """preparing features"""
    def __init__(self, data_file, label_file, groups = 20, docs = 11269, vocabulary = 61188):
        self.data_file = data_file
        self.label_file = label_file
        self.groups = groups
        self.docs = docs
        self.vocabulary = vocabulary
        self.binary_feature = None
        self.tf_feature = None
        self.tfidf_feature = None
        self.label = np.loadtxt(label_file, usecols = (0,))

    def create_features(self):
        train_data = np.loadtxt(self.data_file, usecols = (0,1,2), dtype = np.int32)
        doc_ids  = train_data[:,0] - 1 # make doc_id start from zero
        word_ids = train_data[:,1] - 1 # make word_id start from zero
        word_f   = train_data[:,2]
        indptr = [0]
        dict_helper = {}
        for i in range(len(doc_ids)):
            dict_helper[doc_ids[i]] = i
        for k, v in dict_helper.items():
            indptr.append(v+1)
        #shape = (self.docs, self.vocabulary),
        self.tf_feature = csr_matrix((word_f, word_ids, indptr), dtype = np.float64)

        # convert to binary features
        binary_word_f = np.ones(len(word_f), dtype = np.float64)
        #shape = (self.docs, self.vocabulary)
        self.binary_feature = csr_matrix((binary_word_f, word_ids, indptr), dtype = np.float64)

        # convert to tfidf features
        doc_num = np.zeros(self.vocabulary, dtype = np.uint16)
        sorted_word_ids = np.sort(word_ids)
        for i in sorted_word_ids:
            doc_num[i] += 1
        doc_num = doc_num.astype(np.float64)
        tfidf_word_f = np.zeros(len(word_f), dtype = np.float64)
        temp = np.log(self.docs)
        for i in range(len(word_f)):
            tfidf_word_f[i] = word_f[i] * ( temp - np.log(doc_num[word_ids[i]]) )

        #shape = (self.docs, self.vocabulary),
        self.tfidf_feature = csr_matrix((tfidf_word_f, word_ids, indptr), dtype = np.float64)

def clean_test_data(test_data_file, max_id): # remove the words which don't appear in the training set
    test_data = np.loadtxt(test_data_file, dtype = np.uint16, usecols = (0,1,2))
    word_ids = test_data[:,1] - 1 # make word_id start from zero
    rows_to_be_deleted = []
    for i in range(len(word_ids)):
        if word_ids[i] > max_id:
            rows_to_be_deleted.append(i)
    test_data = np.delete(test_data, tuple(rows_to_be_deleted), axis = 0)
    np.savetxt('cleaned_test.data', test_data)


if __name__ == '__main__':

    folder = '../20news-bydate/matlab/' # don't forget the last slash
    data_file = folder + 'train.data'
    label_file = folder + 'train.label'

    # features in train set
    features = FeatureGenerator(data_file, label_file)
    cpu_time = time.clock()
    features.create_features()
    cpu_time = time.clock() - cpu_time
    logger.info('time used for creating training features: %f', cpu_time)
    logger.info('the shape of tfidf_feature is : (%d, %d)', features.tfidf_feature.shape[0],
        features.tfidf_feature.shape[1])

    test_data_file = './cleaned_test.data'
    #test_data_file = folder + 'test.data'
    test_label_file = folder + 'test.label'
    #clean_test_data(test_data_file, 53974)
    test_features = FeatureGenerator(test_data_file, test_label_file, docs = 7505)
    cpu_time = time.clock()
    test_features.create_features()
    cpu_time = time.clock() - cpu_time
    logger.info('time used for creating testing features: %f', cpu_time)

    cross_validated = True

    if not cross_validated:
        log_param_grid = [
          {'C': [0.05, 0.10, 0.50, 1.0, 5.0 ], 'penalty': ['l1', 'l2']},
        ]
        cpu_time = time.clock()
        log_search = grid_search.GridSearchCV(linear_model.LogisticRegression(), log_param_grid, cv=5,
            scoring='f1_macro')
        log_search.fit(features.tfidf_feature, features.label)
        cpu_time = time.clock() - cpu_time
        logger.info('time used for searching the best parameters for LogisticRegression: %f', cpu_time)
        logger.info('the best parameters for LogisticRegression is : %s', log_search.best_params_)
        logger.info('Grid scores are as below:')
        for params, mean_score, scores in log_search.grid_scores_:
            logger.info("%0.5f (+/-%0.03f) for %s", mean_score, scores.std() * 2, params)
        logger.info('End of grid scores for LogisticRegression')
        best_log = log_search.best_estimator_
        logger.info('the parameters for best LogisticRegression are : %s', best_log.get_params())

    Cs = [0.05, 0.10]
    for c in Cs:
        logger.info('---------------------- penalty = l2, C = %f ----------------------', c)
        best_log = linear_model.LogisticRegression(C = c, penalty = 'l2')
        best_log.fit(features.tfidf_feature, features.label)
        cpu_time = time.clock()
        test_pred = best_log.predict(test_features.tfidf_feature)
        cpu_time = time.clock() - cpu_time
        logger.info('time used for predicting the training data with LogisticRegression: %f', cpu_time)

        log_f1 = metrics.f1_score(test_features.label, test_pred, average='macro')
        logger.info('F_1 score for the test data with best LogisticRegression: %f', log_f1)

    param_grid = [
     {'C': [0.05,], 'penalty': ['l2']},
     {'C': [0.10,], 'penalty': ['l2']}
    ]
    svm_classifiers = [svm.LinearSVC(), svm.LinearSVC()]

    for i in range(len(svm_classifiers)):
        logger.info('-------------------------- i = %d ---------------------------------------', i)
        if not cross_validated:
            cpu_time = time.clock()
            svc_search = grid_search.GridSearchCV(svm_classifiers[i], param_grid[i], cv=5,
                scoring='f1_macro')
            svc_search.fit(features.tfidf_feature, features.label)
            cpu_time = time.clock() - cpu_time
            logger.info('time used for searching the best parameters for svm: %f', cpu_time)
            logger.info('the best parameters for svm is : %s', svc_search.best_params_)
            logger.info('Grid scores are as below:')
            for params, mean_score, scores in svc_search.grid_scores_:
                logger.info("%0.5f (+/-%0.03f) for %s", mean_score, scores.std() * 2, params)
            logger.info('End of grid scores for svm')
            best_svc = svc_search.best_estimator_
            logger.info('the parameters for best svm are : %s', best_svc.get_params())

        best_svc = svm.LinearSVC(C = Cs[i], penalty = 'l2')
        best_svc.fit(features.tfidf_feature, features.label)
        cpu_time = time.clock()
        test_pred = best_svc.predict(test_features.tfidf_feature)
        cpu_time = time.clock() - cpu_time
        logger.info('time used for predicting the training data with svm: %f', cpu_time)

        svc_f1 = metrics.f1_score(test_features.label, test_pred, average='macro')
        logger.info('F_1 score for the test data with best svm: %f', svc_f1)



