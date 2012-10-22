import util;
import os;

class PRelFile(util.KeyMapMap):
    def __init__(self, path):
        util.KeyMapMap.__init__(self, 0);
        self._load(path);

    def _load(self, path):
        f = open(path);
        lines = map(str.strip, f.readlines());
        for line in lines:
            tokens = line.split();
            qid, docid, score = tokens[:3];
            qid = int(qid);
            self.add(qid, docid, score);
            score = int(score)


class QRelFile(util.KeyMapMap):
    def __init__(self, path):
        util.KeyMapMap.__init__(self, 0);
        self._load(path);

    def _load(self, path):
        f = open(path);
        lines = map(str.strip, f.readlines());
        for line in lines:
            tokens = line.split();
            qid, nothing, docid, score = tokens[:4];
            qid = int(qid);
            if docid == '1':
                print line;
            score = int(score)
            self.add(qid, docid, score);

    def store(self, path):
        f = open(path, 'w');
        for qid in self.keys():
            for docid, score in self._data[qid].items():
                f.write('%d 0 %s %d\n' % (qid, docid, score));
        f.close();

def balance_judge(qrel):
    for topic, topic_judge in qrel.items():
        rel_topic_judge, irrel_topic_judge = [], []
        for doc, judge in topic_judge.items():
            if judge > 0:
                rel_topic_judge.append((doc,judge))
            else:
                irrel_topic_judge.append((doc, judge))
        c = min(len(rel_topic_judge), len(irrel_topic_judge))
        new_topic_judge = {}
        for i in xrange(c):
            doc, judge = rel_topic_judge[i]
            new_topic_judge[doc] = judge
            doc, judge = irrel_topic_judge[i]
            new_topic_judge[doc] = judge
        qrel[topic] = new_topic_judge


def main():
    pass;

if __name__ == '__main__':
    main()















