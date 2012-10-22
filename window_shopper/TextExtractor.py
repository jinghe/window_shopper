from JudgeFile import QRelFile;
from Index import Index;
from TRECTopics import StandardFormat;
from Corpus import TRECReader
import fastmap;
from TextUtil import *;

import bsddb;
import os;
import subprocess;
import sys;
from multiprocessing import Pool;
import numpy as np;


suffixes = ['html', 'text', 'title'];
extract_script = 'extract-chunks-from-page.py';

def extract_text(docno, index_path, collection_type='html'):
    '''
        collection_type: 'html' or 'text'
    '''
    text = '';
    try:
        index = Index(index_path);
        content = index.get_doc_content(docno);
        if collection_type == 'html':
            html_path, text_path, title_path = map(lambda suffix: '%s.%s' % (docno, suffix), suffixes);
            f = open(html_path, 'w')
            f.write(content)
            f.close()
            subprocess.call(['python', extract_script, html_path, text_path, title_path])
            #title_f = open(title_path)
            # first line is the title
            #text = ' '.join(map(str.strip, title_f.readlines())) + '\n'
            text_f = open(text_path)
            text += ''.join(text_f.readlines())
            os.remove(html_path);
            os.remove(text_path);
            os.remove(title_path);
        elif collection_type == 'text': text = content
        
    except Exception, e:
        sys.stderr.write('error at docno %s\n' % docno);
    return text;

def is_cluewebB(docno):
    col_name = docno.split('-')[1];
    no = int(col_name[4:]);
    if col_name.startswith('en00') and no <= 11:
        return True;
    if col_name.startswith('enwp') and no <= 3:
        return True;
    return False;

def exe_extract_text(judge_path, index_path, out_path, collection_type = 'html'):
    '''
        extract texts of docs in qrel from an index, and store them in out_path in standard trec format
    '''
    import Corpus
    judge_file = QRelFile(judge_path);
    docnos = judge_file.key2s();
    print 'doc number:', len(docnos);
    writer = Corpus.TRECWriter(out_path);
    for docno in docnos:
        text = extract_text(docno, index_path, collection_type)
        writer.write(Corpus.Document(docno, text))

def test_extract_text(judge_path, index_path, collection_type):
    judge_file = QRelFile(judge_path);
    docnos = judge_file.key2s();
    print 'doc number:', len(docnos);
    for docno in docnos[:1]:
        text = extract_text(docno, index_path, collection_type);
        print text
        print '-' * 20

def exe_extract_words(article_path, topic_path, out_word_path):
    term_set = set()

    topics = StandardFormat().read(topic_path)
    for topic_id, topic_string in topics.items():
        topic = complete_text_work(topic_string)
        for token in topic.tokens:
            term_set.add(token)

    reader = TRECReader()
    reader.open(article_path)
    doc = reader.next()
    while doc:
        print doc.ID, len(term_set)
        text = complete_text_work(doc.text)
        for token in text.tokens:
            term_set.add(token)
        doc = reader.next()

    print 'writing.....'
    word_list_file = open(out_word_path, 'w');
    words = list(term_set)
    words.sort();
    map(lambda word:word_list_file.write('%s\n' % word), words);
    word_list_file.close();

if __name__ == '__main__':
    option = sys.argv[1];
    argv = sys.argv[2:]
    if option == '--extract-text':
        exe_extract_text(*argv);
    elif option == '--test-extract-text':
        test_extract_text(*argv);
    elif option == '--extract-word':
        exe_extract_words(*argv);
    else:
        print 'error param';
    

