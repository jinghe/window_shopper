from sklearn.externals import joblib
from TextUtil import *
from FeatureExtractor import *
from gradient_boosting_ranker import *
from ParagraphSegmenter import ParagraphSegmenter
import numpy as np

class ParagraphScorer:
    def __init__(self, model_path, feature_extractor):
        self.model = joblib.load(model_path)
        self.feature_extractor = feature_extractor

    def score(self, query, doc, paragraph):
        '''
            return a score value (float) for a paragraph
        '''
        feature_values = self.feature_extractor.extract(query, doc, paragraph)
        X = np.array([feature_values])
        y = self.model.predict(X)[0]
        return y[0]

    def rank(self, query, doc, paragraphs):
        '''
            return a list of (score, paragraph) sorted by the scores
        '''
        score_paragraph_list = []
        for paragraph in paragraphs:
            score_paragraph_list.append((self.score(query, doc, paragraph), paragraph))
        score_paragraph_list.sort(reverse = True)
        return score_paragraph_list

query_model_factory = DocumentModelFactory(FlatTFFactory(), DictIDFFactory())
query_text_chain = TextChain([TextTokenizer(wordpunct_tokenize), TextTokenNormalizer(), TextStopRemover(), TextStemmer(stemmer), TextModeler(query_model_factory)])

doc_model_factory = DocumentModelFactory(MaxNormTFFactory(), NoIDFFactory())
doc_text_chain = TextChain([TextTokenizer(wordpunct_tokenize), TextTokenNormalizer(), TextStopRemover(), TextStemmer(stemmer), TextModeler(doc_model_factory)])
build_model_chain = TextChain([TextModeler(doc_model_factory)])

scorer = CosTextScorer()
feature_extractor = MultipleFeatureExtractor([QueryFeatureExtractor(), ParagraphFeatureExtractor(), FidelityExtractor(scorer), QDRelevanceExtractor([scorer]), QSRelevanceExtractor([scorer])])

def do_score_paragraph(query_path, doc_path, paragraph_path, model_path):
    scorer = CosTextScorer()
    paragraph_scorer = ParagraphScorer(model_path, feature_extractor)
    query_string = open(query_path).read()
    query = complete_text_work(query_string, query_text_chain)
    doc_string = open(doc_path).read()
    doc = complete_text_work(doc_string, doc_text_chain)
    paragraph_string = open(paragraph_path).read()
    paragraph = complete_text_work(paragraph_string, doc_text_chain)
    print paragraph_scorer.score(query, doc, paragraph)

def do_rank_paragraph(query_path, doc_path, model_path, paragraph_length, paragraph_inc):
    paragraph_length = int(paragraph_length)
    paragraph_inc = int(paragraph_inc)

    query_string = open(query_path).read()
    query = complete_text_work(query_string, query_text_chain)
    doc_string = open(doc_path).read()
    doc = complete_text_work(doc_string, doc_text_chain)

    segmenter = ParagraphSegmenter(paragraph_length, paragraph_inc)
    paragraphs = segmenter.segment_object(query, doc)
    paragraphs = map(lambda paragraph: build_model_chain.work(paragraph), paragraphs)

    paragraph_scorer = ParagraphScorer(model_path, feature_extractor)
    score_paragraphs = paragraph_scorer.rank(query, doc, paragraphs)
    for score, paragraph in score_paragraphs:
        print score, ' '.join(paragraph.tokens)
        print 

if __name__ == '__main__':
    import sys
    option = sys.argv[1]
    argv = sys.argv[2:]
    if option == '--score-paragraph':
        do_score_paragraph(*argv)
    elif option == '--rank-paragraph':
        do_rank_paragraph(*argv)




