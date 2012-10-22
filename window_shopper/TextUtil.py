from nltk.tokenize import wordpunct_tokenize, sent_tokenize;
from nltk.stem.snowball import EnglishStemmer
import numpy as np
from Index import *

stemmer = EnglishStemmer()

class TextPiece:
    '''
        class of a text piece
    '''
    def __init__(self, text):
        self.text = text.decode('utf8', errors='ignore');

class TextProcessor:
    '''
        abstract class for a text processor;
'''
    def work(self, text_piece):
        pass;

class TextChain(TextProcessor):
    def __init__(self, workers):
        self.workers = workers;

    def work(self, text_piece):
        for worker in self.workers:
            worker.work(text_piece);
        return text_piece

class TextTokenizer(TextProcessor):
    '''
        create text.tokens field
    '''
    def __init__(self, tokenize_func):
        self.tokenize_func = tokenize_func;

    def work(self, text_piece):
        text_piece.tokens = self.tokenize_func(text_piece.text);
        return text_piece

class TextTokenNormalizer(TextProcessor):
    '''
        normalize to lower case
    '''
    def work(self, text_piece):
        text_piece.tokens = map(lambda token: token.lower(), text_piece.tokens);

class TextStopRemover(TextProcessor):
    '''
        remove the stop words;
    '''
    def work(self, text_piece):
        text_piece.tokens = filter(lambda token: not STOP_SET.__contains__(token.lower()), text_piece.tokens);

class TextStemmer:
    '''
        stem
    '''
    def __init__(self, stemmer):
        self.stemmer = stemmer;

    def work(self, text_piece):
        text_piece.tokens = map(lambda token: self.stemmer.stem(token), text_piece.tokens);


class BigramCounter:
    def __init__(self, index_path):
        self.index = Index(index_path)
        self.cf = self.index.index_stats()[2]

    def work(self, text_piece):
        counts = {}
        if len(text_piece.tokens) < 2:
            return 
        for i in xrange(len(text_piece.tokens) - 1):
            token_pair_string = '#1(%s)' % (' '.join(text_piece.tokens[i: i + 2]))
            counts[token_pair_string] = self.index.get_count(token_pair_string)
        token_set = list(set(text_piece.tokens))
        for i in xrange(len(token_set) - 1):
            token1 = token_set[i]
            for j in xrange(i+1, len(token_set)):
                token2 = token_set[j]
                token_pair_string = '#uw8(%s %s)' % (token1, token2);
                counts[token_pair_string] = self.index.get_count(token_pair_string)
        text_piece.bigram_counts = counts
        text_piece.bigram_cf = self.cf

            

class IDFDictionary:
    def __init__(self, path):
        self.data = {};
        f = open(path);
        lines = f.readlines();
        for line in lines:
            term, idf = line.split();
            self.data[term] = float(idf);
        f.close();
        self.stemmer = stemmer

    def get(self, term, normalized = True):
        if not normalized:
            term = self.stemmer.stem(term.lower())
        if self.data.has_key(term):
            return self.data[term]
        elif (any(map(lambda c: c < 'a' and c > 'z', term)) and any(map(lambda c: c < '0' and c > '9'), term)) or STOP_SET.__contains__(term):
            return 0
        else:
            return 10.0


class TextModeler:
    '''
        create a lm field: language model or vector space model
    '''
    def __init__(self, model_factory):
        self.model_factory = model_factory;

    def work(self, text_piece):
        text_piece.lm = self.model_factory.build(text_piece.tokens);
        return text_piece

class FlatTFFactory:
    def build(self, tokens):
        tfs = {}
        map(lambda token: tfs.__setitem__(token, tfs.get(token,0) + 1), tokens);
        return tfs

class LogTFFactory(dict):
    def build(self, tokens):
        tfs = FlatTFFactory().build(tokens) 
        for token, flatTF in tfs.items():
            tfs[token] = 1 + np.log(flatTF)
        return tfs

class MaxNormTFFactory(dict):
    def build(self, tokens):
        tfs = FlatTFFactory().build(tokens) 
        max_flat_tf = max(tfs.values())
        for token, flatTF in tfs.items():
            tfs[token] = 0.5 + 0.5 * flatTF / max_flat_tf
        return tfs

class DictIDFFactory:
    def get(self, token):
        return IDF_DICT.get(token)

class NoIDFFactory:
    def get(self, token):
        return 1.0

class DocumentModelFactory:
    def __init__(self, tf_factory, idf_factory):
        self.tf_factory = tf_factory
        self.idf_factory = idf_factory
    
    def build(self, tokens):
        token_tf = {};
        doc_model = {};
        tfs = self.tf_factory.build(tokens)
        for token, tf in tfs.items():
            idf = self.idf_factory.get(token);
            if idf:
                doc_model[token] = tf * idf;
        return doc_model;

class CosTextScorer:
    def score(self, topic, doc):
        model1, model2 = topic.lm, doc.lm
        return [self.dot_product(model1, model2)/(self.norm(model1) * self.norm(model2))]

    def dot_product(self, model1, model2):
        keys = set(model1.keys());
        keys = keys.intersection(model2.keys());
        val = 0;
        for key in keys:
            val += model1[key] * model2[key];
        return val;

    def norm(self, model):
        norm_val = np.sqrt(sum(map(lambda val: val ** 2, model.values())));
        return norm_val;


class BigramScorer:
    def __init__(self, mu):
        self.mu = mu;

    def score(self, query, doc):
        if len(query.tokens) == 1:
            return [1] * 2;
        match_scores = []
        token_pos = self.build_token_pos(doc, set(query.tokens))
        match_scores += self.match_ordered(query, doc, token_pos)
        match_scores += self.match_unordered(query, doc, token_pos)
        return match_scores

    def build_token_pos(self, doc, query_token_set):
        token_pos = {}
        for i in xrange(len(doc.tokens)):
            token = doc.tokens[i]
            if query_token_set.__contains__(token):
                if not token_pos.has_key(token):
                    token_pos[token] = [i]
                else:
                    token_pos[token].append(i)
        return token_pos

    def match_ordered(self, query, doc, token_pos):
        scores = []
        for i in xrange(len(query.tokens) - 1):
            token1 = query.tokens[i]
            token2 = query.tokens[i+1]
            match_num = 0
            if token_pos.has_key(token1) and token_pos.has_key(token2):
                positions1, positions2 = token_pos[token1], token_pos[token2]
                index1, index2 = 0, 0
                while index1 < len(positions1) and index2 < len(positions2):
                    pos1_next = positions1[index1] + 1
                    pos2 = positions2[index2]
                    if pos1_next == pos2:
                        match_num += 1
                        index1 += 1
                        index2 += 1
                    elif pos1_next < pos2:
                        index1 += 1
                    else:
                        index2 += 1
            query_string = '#1(%s %s)' % (token1, token2)
            scores.append(self.score_bigram(match_num, len(doc.tokens), query.bigram_counts[query_string], query.bigram_cf));
        return [np.mean(scores)]

    def match_unordered(self, query, doc, token_pos):
        scores = []
        doc_len = len(doc.tokens)
        token_set = list(set(query.tokens))
        for i in xrange(len(token_set)-1):
            token1 = token_set[i]
            for j in xrange(i+1, len(token_set)):
                token2 = token_set[j]
                match_num = 0
                if token_pos.has_key(token1) and token_pos.has_key(token2):
                    positions1, positions2 = token_pos[token1], token_pos[token2]
                    index1, index2 = 0, 0
                    while index1 < len(positions1) and index2 < len(positions2):
                        pos1 = positions1[index1]
                        pos2 = positions2[index2]
                        if np.abs(pos1 - pos2) <= 8:
                            match_num += 1
                        if pos1 < pos2:
                            index1 += 1
                        else:
                            index2 += 1
                query_string = '#uw8(%s %s)' % (token1, token2)
                scores.append(self.score_bigram(match_num, len(doc.tokens), query.bigram_counts[query_string], query.bigram_cf));
        return [np.mean(scores)]

    def score_bigram(self, tf, doc_len, cf, coll_len):
        if cf == 0:
            cf = 1
        coll_ratio = cf / float(coll_len)
        prob = (tf + coll_ratio * self.mu) / (doc_len + self.mu)
        #print tf, doc_len, coll_ratio, self.mu, prob
        return np.log(prob)

STOP_SET = set(map(lambda line: line.strip(), open('stoplist.dft').readlines()));
IDF_DICT = IDFDictionary('enidf.norm.txt')
#COMPLETE_TEXT_CHAIN = TextChain([TextTokenizer(wordpunct_tokenize), TextTokenNormalizer(), TextStopRemover(), TextStemmer(stemmer), TextModeler(model_factory)])
#MODELER_TEXT_CHAIN = TextChain([TextModeler(model_factory)])

def complete_text_work(text, text_chain):
    text_object = TextPiece(text)
    text_chain.work(text_object)
    return text_object

def test_cos():
    text1 = complete_text_work("Moore's Law sustained momentum despite worries")
    text2 = complete_text_work("Moore's law observation Despite popular misconception")
    print text1.lm
    print text2.lm
    scorer = CosTextScorer()
    print scorer.score(text1.lm, text2.lm)
    print scorer.score(text1.lm, text1.lm)
    print scorer.score(text2.lm, text1.lm)
    print scorer.score(text2.lm, text2.lm)

if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    test_cos(*argv)
