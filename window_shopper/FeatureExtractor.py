from JudgeFile import QRelFile;
from Index import Index;
from TRECTopics import StandardFormat;
import fastmap;
from scipy import stats

from TextUtil import *

task_num = 4;

class ExtractorError(Exception):
    def __init__(self, msg):
        self.msg = msg;

    def __str__(self):
        return self.msg;

'''
require:
    1) tokenized;
    2) stop words removed;
    3) stemmed;
    4) sentence tokenized;
result:
    text_piece.significant_terms;
'''
class SignificantTermDetector:
    def work(self, text_piece):
        sentence_num = len(text_piece.sentences);
        term_counts = {};
        for token in text_piece.tokens:
            term_counts.__setitem__(token, term_counts.get(token, 0) + 1);
        term_count_list = term_counts.items(); 
        term_count_list.sort(reverse=True, key=lambda term_count: term_count[1]);

        term_num = 7;
        if sentence_num > 40:
            term_num += .1 * (sentence_num - 40);
        elif sentence_num < 25:
            term_num -= .1 * (25 - sentence_num);
        text_piece.significant_terms = map(lambda term_count: term_count[0], term_count_list[:term_num]);


class FeatureExtractorInterface:
    def extract(self, topic, doc, paragraph):
        pass

class QueryFeatureExtractor(FeatureExtractorInterface):
    def extract(self, topic, doc = None, window = None):
        topic_term_num = len(topic.tokens);
        idfs = [];
        for token in topic.tokens:
            idf = IDF_DICT.get(token, True)
            if idf:
                idfs.append(idf)
        return topic_term_num, np.exp(topic_term_num), np.log(topic_term_num), np.mean(idfs), stats.gmean(idfs), sum(idfs);

class ParagraphFeatureExtractor(FeatureExtractorInterface):
    def extract(self, topic, doc, window):
        idfs = [];
        for token in window.tokens:
            idf = IDF_DICT.get(token, True)
            if idf:
                idfs.append(idf)        
        if len(idfs) == 0:
            raise ExtractorError('no valid token in the window "%s"' % window.text.encode('utf8'));
        return np.mean(idfs), stats.gmean(idfs);

class DocumentExtractor(FeatureExtractorInterface):
    def extract(self, topic, doc, window = None):
        term_num = len(doc.tokens)
        idfs = [];
        for token in doc.tokens:
            idf = IDF_DICT.get(token, True)
            if idf:
                idfs.append(idf)
        return term_num, np.log(term_num), np.mean(idfs), stats.gmean(idfs), sum(idfs);

class FidelityExtractor(FeatureExtractorInterface):
    def __init__(self, scorer):
        self.scorer = scorer;

    def extract(self, topic, doc, window):
        return self.scorer.score(doc, window);

class QDRelevanceExtractor(FeatureExtractorInterface):
    def __init__(self, scorers):
        self.scorers = scorers;

    def extract(self,topic, doc, window):
        scores = []
        for scorer in self.scorers:
            subscores = scorer.score(topic, doc)
            scores += subscores
        return scores

class QSRelevanceExtractor(FeatureExtractorInterface):
    def __init__(self, scorers):
        self.scorers = scorers;

    def extract(self,topic, doc, window):
        scores = []
        for scorer in self.scorers:
            subscores = scorer.score(topic, window)
            scores += subscores
        return scores

class MultipleFeatureExtractor(FeatureExtractorInterface):
    def __init__(self, extractors):
        self.extractors = extractors

    def extract(self, topic, doc, paragraph):
        feature_values = []
        for extractor in self.extractors:
            feature_values += extractor.extract(topic, doc, paragraph)
#print extractor.__class__, feature_values
        return feature_values


