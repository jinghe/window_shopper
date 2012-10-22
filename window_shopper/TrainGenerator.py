from JudgeFile import QRelFile;
from JudgeFile import balance_judge;
from Index import Index;
from TRECTopics import StandardFormat;
from TextUtil import *;
from TextExtractor import *;
from ParagraphSegmenter import ParagraphSegmenter
from FeatureExtractor import *
from RelevancePredictor import qd_rel_learner_extractor
from RelevancePredictor import qs_rel_learner_extractor

import numpy as np;
import time;
import sys;
import bsddb;
from nltk.stem.snowball import EnglishStemmer
from multiprocessing import Pool;
import traceback;

stemmer = EnglishStemmer();
stop_path = 'data/stoplist.dft';

query_model_factory = DocumentModelFactory(FlatTFFactory(), DictIDFFactory())
query_text_chain = TextChain([TextTokenizer(wordpunct_tokenize), TextTokenNormalizer(), TextStopRemover(), TextStemmer(stemmer), TextModeler(query_model_factory)])

doc_model_factory = DocumentModelFactory(MaxNormTFFactory(), NoIDFFactory())
doc_text_chain = TextChain([TextTokenizer(wordpunct_tokenize), TextTokenNormalizer(), TextStopRemover(), TextStemmer(stemmer), TextModeler(doc_model_factory)])
build_model_chain = TextChain([TextModeler(doc_model_factory)])


'''
aggregate by average of k-nearest neighbors' distances
'''
class Aggregator:
    def __init__(self, K):
        self.K = K;

    def aggregate(self, values):
        values.sort(reverse = True);
        return np.mean(values[:self.K]);

class Document:
    def __init__(self, docno, doc, windows, rel):
        self.docno = docno;
        self.windows = windows;
        self.doc = doc;
        self.rel = rel;

           
def build_train(topic_id):
    try:
        print 'topic:', topic_id;
        docs = [];
        if not topics.has_key(topic_id):
            return docs;
        topic_str = topics[topic_id];
        for docno, rel in judge_file[topic_id].items():
            if not is_cluewebB(docno):
                continue;
            try:
                windows = map(lambda sentence: TextPiece(sentence), window_db[docno].split('\n'));
                docs.append(Document(docno, TextPiece(doc_db[docno]), windows, int(rel)));
            except Exception,e:
                sys.stderr.write(str(traceback.format_exc()));
                sys.stderr.write('%s %s: error at %s\n' % (str(e.__class__), str(e),docno));
                sys.stderr.write('-' * 100 + '\n');
                sys.exit(-1);
        ranker.rank(topic_str, docs);
    except Exception as e:
        print traceback.format_exc();
        sys.exit(-1);
    return docs;
                
  
def test_stat(argv):
    stat_path = argv[0];
    t0 = time.time();
    print time.time() - t0;

def gen_field(writer, docno, sentence_lines, score_id, sentence_num):
    score_sentences = [];
    for sentence_line in sentence_lines:
        pos = sentence_line.find(':');
        scores = sentence_line[:pos];
        sentence = sentence_line[pos+1:];
        score = float(scores.split(',')[score_id]);
        score_sentences.append((score, sentence));
    score_sentences.sort();
    snippet = ' |||| '.join(map(lambda score_sentence: score_sentence[1], score_sentences[:sentence_num]));
    writer.write('>>>>docno:%s\n' % docno);
    writer.write('>>>>desc:%s\n' % snippet);
    writer.write('----END-OF-RECORD----\n');

def exe_gen_field(argv):
    snippet_path = argv[0];
    average_num = int(argv[1]);
    sentence_num = int(argv[2]);
    out_path = argv[3];
    score_id = K_options.index(average_num);

    f = open(snippet_path);
    writer = open(out_path, 'w');
    line = f.readline();
    docno = 0;
    sentence_lines = [];
    while line:
        line = line.strip();
        if line.startswith('docno:'):
            if docno:
                gen_field(writer, docno, sentence_lines, score_id, sentence_num);
                sentence_lines = [];
            docno = line[6:];
        elif line.startswith('topic_id:'):
            pass;
        else:
            sentence_lines.append(line);
        line = f.readline();
    writer.close();
    f.close();

def exe_view_train(argv):
    train_path, window_path, doc_path = argv;
    from Learner import TrainFile;
    train_file = TrainFile();
    train_file.load(train_path);
    window_db = bsddb.hashopen(window_path);
    doc_db = bsddb.hashopen(doc_path);

    num = 1000;
    key = train_file.keys()[num];
    qid, docno, rel, sid = key.split();
    doc_text = doc_db[docno];
    print qid, docno;
    print doc_text;
    print '=' * 50;
    windows = window_db[docno].split('\n'); 
    window_scores = [];
    for key in train_file.keys()[num:]:
        qid, curr_docno, rel, sid = key.split();
        if curr_docno <> docno:
            break;
        window_text = windows[int(sid)]; 
        value = train_file[key];
        window_scores.append((value, window_text));
    window_scores.sort();
    for score, window_text in window_scores:
        print score, window_text;

class ParagraphScorer:
    def score(self, topic, doc, paragraph):
        return CosTextScorer().score(topic, paragraph)

class TrainGenerator:
    def __init__(self, paragraph_segmenter, feature_extractor, paragraph_scorer):
        self.paragraph_segmenter = paragraph_segmenter
        self.feature_extractor = feature_extractor
        self.paragraph_scorer = paragraph_scorer

    def run(self, doc_path, topic_path, judge_path, out_path, doc_limit):
        qrel_file = QRelFile(judge_path);
        balance_judge(qrel_file)
        doc_topic_dict = self.build_doc_topic_dict(qrel_file)
        topics = StandardFormat().read(topic_path);
        topics = self.parse_topics(topics)
        writer = open(out_path, 'w')

        doc_reader = TRECReader()
        doc_reader.open(doc_path)
        doc = doc_reader.next()
        doc_count = 0
        paragraph_count = 0
        t0 = time.time()
        while doc and doc_count < doc_limit:
            doc_string = doc.text
            doc_id = doc.ID
            doc_object = complete_text_work(doc_string, doc_text_chain)
            if doc_topic_dict.has_key(doc_id):
                for topic_id in doc_topic_dict[doc_id]:
                    topic = topics[topic_id];
                    paragraphs = self.paragraph_segmenter.segment_object(topic, doc_object)
                    for paragraph in paragraphs:
                        build_model_chain.work(paragraph)
                        feature_values = self.feature_extractor.extract(topic, doc_object, paragraph)
                        rel = qrel_file[topic_id][doc_id]
                        target = self.paragraph_scorer.score(topic, doc_object, paragraph)[0]
                        writer.write('%d %s %d %f %s\n' % (topic_id, doc.ID, rel, target, ' '.join(map(lambda value: str(value), feature_values))))
                        paragraph_count += 1
            doc = doc_reader.next()
            doc_count += 1
            if doc_count % 500 == 0:
                print doc_count, paragraph_count, time.time() - t0
        writer.close()

    def build_doc_topic_dict(self, qrel_file):
        '''
            build a dict: doc_id -> [topic_id,...] 
        '''
        doc_topic_dict = {}
        for topic_id, topic_judges in qrel_file.items():
            for doc_id, judge in topic_judges.items():
                if not doc_topic_dict.has_key(doc_id):
                    doc_topic_dict[doc_id] = [topic_id]
                else:
                    doc_topic_dict[doc_id].append(topic_id)
        return doc_topic_dict

    def parse_topics(self,topics):
        '''
            parse topics 
        '''
        new_topics = {}
        for topic_id, topic_string in topics.items():
            new_topics[topic_id] = complete_text_work(topic_string, query_text_chain)
        return new_topics


def do_build_train(topic_path, judge_path, doc_path, out_path, paragraph_length, paragraph_increment, doc_limit):
    #1.1 prepare workers;
        #1.2 prepare the segmenter
    paragraph_segmenter = ParagraphSegmenter(int(paragraph_length), int(paragraph_increment))

    #1.3 prepare the feature extractors
    scorer = CosTextScorer()
    feature_extractor = MultipleFeatureExtractor([QueryFeatureExtractor(), ParagraphFeatureExtractor(), FidelityExtractor(scorer), QDRelevanceExtractor([scorer]), QSRelevanceExtractor([scorer])])

    #2. build the training data;
    train_generator = TrainGenerator(paragraph_segmenter, feature_extractor, ParagraphScorer())
    train_generator.run(doc_path, topic_path, judge_path, out_path, int(doc_limit))

def do_gen_simulation(out_path):
    query_num = 100
    doc_num = 100
    sen_num = 5
    writer = open(out_path, 'w')
    for i in xrange(query_num):
        queryno = i+1
        for j in xrange(doc_num):
            docno = i * query_num + j
            #rel = np.random.binomial(1, .15)
            rel = np.random.binomial(1, .15)
            if rel:
                qd_sim = np.random.normal(0.6, 0.35)
            else:
                qd_sim = np.random.normal(0.3, 0.35)
            for k in xrange(sen_num):
                #qs_sim = np.random.randint(5)
                #qs_sim = np.random.randint(5)
                qs_sim = np.random.random()
                ds_sim = np.random.random()
                #if rel:
                    #diff = np.random.normal(0.2, 0.1)
                #else:
                    #diff = np.random.normal(0.5, 0.1)
                #pos_nev = np.random.randint(2)
                #if pos_nev:
                    #ds_sim = qs_sim + diff
                #else:
                    #ds_sim = qs_sim - diff
                writer.write('%d %d %d %f %f %f %f\n' % (queryno, docno, rel, qs_sim, qs_sim, ds_sim, qd_sim))
    writer.close()

if __name__ == '__main__':
    option = sys.argv[1];
    argv = sys.argv[2:]
    if option == '--gen-train':
        do_build_train(*argv);
    elif option == '--test-stat':
        test_stat(*argv);
    elif option == '--gen-field':
        exe_gen_field(*argv);
    elif option == '--view-train':
        exe_view_train(*argv);
    elif option == '--simulate':
        do_gen_simulation(*argv)
        
