window_shopper
==============

Finding the best window around a keyword match in a document.

Distributed under the same license as the Lemur Toolkit.

Dependent Packages
------------------

* Numpy
* Scipy
* scikit-learn
* nltk 2.0.1

Re-creating the data/ folder
----------------------------

The system needs a dict folder with the following data:

* stoplist.dft: a list of stop words
* idf.norm.txt: IDF score for a list of terms

Basic Usage
-----------
python SnippetGenerator --score-paragraph <query-path> <doc-path> <paragraph-path> <model-path>
python SnippetGenerator --rank-paragraph <query-path> <doc-path> <model-path> <paragraph-len> <paragraph-inc>

