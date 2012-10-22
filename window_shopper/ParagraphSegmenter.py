from TextUtil import *
from JudgeFile import *
from Corpus import *
from TRECTopics import *

class Paragraph(TextPiece):
    def __init__(self, doc, begin, length):
        self.tokens = doc.tokens[begin: begin+length]
        self.text = ' '.join(self.tokens)

class ParagraphSegmenter:
    def __init__(self, length, increment):
        self.length = length
        self.increment = increment

    def segment_string(self, query_string, doc_string):
        query = complete_text_work(query_string)
        query_term_set = set(query.tokens)
        doc = complete_text_work(doc_string)
        print ' '.join(doc.tokens)
        print
        return self.segment_object(query, doc)

    def segment_object(self, query, doc):
        paragraphs = []
        doc_length = len(doc.tokens)
        for begin in xrange(0, doc_length, self.increment):
            paragraphs.append(Paragraph(doc, begin, min(self.length, doc_length - begin)))
        paragraphs = self.filter_paragraph(query, paragraphs)
        return paragraphs

    def filter_paragraph(self, query, paragraphs):
        query_term_set = set(query.tokens)
        valid_paragraphs = []
        for paragraph in paragraphs:
            valid = False
            for token in paragraph.tokens:
                if query_term_set.__contains__(token):
                    valid = True
                    break
            if valid :
                valid_paragraphs.append(paragraph)
        return valid_paragraphs

def do_segment_paragraph(topic_path, judge_path, text_path, out_path):
    print 'loading qrel, topics......'
    judge_file = QRelFile(judge_path);
    topics = StandardFormat().read(topic_path);
    doc_topic_dict = {}
    for topic_id, topic_judges in judge_file.items():
        for doc_id, judge in topic_judges.items():
            if not doc_topic_dict.has_key(doc_id):
                doc_topic_dict[doc_id] = [topic_id]
            else:
                doc_topic_dict[doc_id].append(topic_id)

    print 'segmenting......'
    corpus_reader = TRECReader()
    corpus_reader.open(text_path)
    doc = corpus_reader.next()
    segmenter = ParagraphSegmenter()
    writer = TRECWriter(out_path)
    while doc:
        print doc.ID
        for topic_id in doc_topic_dict[doc.ID]:
            topic_text = topics[topic_id]
            doc_text = doc.text
            paragraphs = segmenter.segment(topic_text, doc_text, 100, 20)
            text = '<topic_ID>%s</topic_ID>\n' % (topic_id)
            for paragraph in paragraphs:
                paragraph_text = '<paragraph>%s</paragraph>' % (' '.join(paragraph.doc.tokens[paragraph.begin: paragraph.begin+paragraph.length]))
                text += paragraph_text.replace('\n',' ') + '\n'
            new_doc = Document(doc.ID, text)
            writer.write(new_doc)
        doc = corpus_reader.next()

def test_segment_paragraph(query_path, doc_path, length, inc):
    query_string = open(query_path).read()
    doc_string = open(doc_path).read()
    segmenter = ParagraphSegmenter(int(length), int(inc))
    paragraphs = segmenter.segment_string(query_string, doc_string)
    for paragraph in paragraphs:
        print ' '.join(paragraph.tokens)
        print
             

if __name__ == '__main__':
    import sys
    option = sys.argv[1]
    argv = sys.argv[2:]
    if option == '--run':
        do_segment_paragraph(*argv)
    elif option == '--test':
        test_segment_paragraph(*argv)


        
       
