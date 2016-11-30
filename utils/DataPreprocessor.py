import numpy as np
import glob
import os
import sys

from config import MAX_WORD_LEN

SYMB_BEGIN = "@begin"
SYMB_END = "@end"

class Data:

    def __init__(self, dictionary, num_entities, training, validation, test):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1])
        self.num_entities = num_entities
        self.inv_dictionary = {v:k for k,v in dictionary[0].items()}

class DataPreprocessor:

    def preprocess(self, question_dir, no_training_set=False, use_chars=True):
        """
        preprocess all data into a standalone Data object.
        the training set will be left out (to save debugging time) when no_training_set is True.
        """
        vocab_f = os.path.join(question_dir,"vocab.txt")
        word_dictionary, char_dictionary, num_entities = \
                self.make_dictionary(question_dir, vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)
        if no_training_set:
            training = None
        else:
            print "preparing training data ..."
            training = self.parse_all_files(question_dir + "/training", dictionary, use_chars)
        print "preparing validation data ..."
        validation = self.parse_all_files(question_dir + "/validation", dictionary, use_chars)
        print "preparing test data ..."
        test = self.parse_all_files(question_dir + "/test", dictionary, use_chars)

        data = Data(dictionary, num_entities, training, validation, test)
        return data

    def make_dictionary(self, question_dir, vocab_file):

        if os.path.exists(vocab_file):
            print "loading vocabularies from " + vocab_file + " ..."
            vocabularies = map(lambda x:x.strip(), open(vocab_file).readlines())
        else:
            print "no " + vocab_file + " found, constructing the vocabulary list ..."

            fnames = []
            fnames += glob.glob(question_dir + "/test/*.question")
            fnames += glob.glob(question_dir + "/validation/*.question")
            fnames += glob.glob(question_dir + "/training/*.question")

            vocab_set = set()
            n = 0.
            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline().split()
                fp.readline()
                query = fp.readline().split()
                fp.close()

                vocab_set |= set(document) | set(query)

                # show progress
                n += 1
                if n % 10000 == 0:
                    print '%3d%%' % int(100*n/len(fnames))

            entities = set(e for e in vocab_set if e.startswith('@entity'))

            # @placehoder, @begin and @end are included in the vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)

            vocabularies = list(entities)+list(tokens)

            print "writing vocabularies to " + vocab_file + " ..."
            vocab_fp = open(vocab_file, "w")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

        vocab_size = len(vocabularies)
        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocabularies if v.startswith('@entity')])
        print "vocab_size = %d" % vocab_size
        print "num characters = %d" % len(char_set)
        print "%d anonymoused entities" % num_entities
        print "%d other tokens (including @placeholder, %s and %s)" % (
                vocab_size-num_entities, SYMB_BEGIN, SYMB_END)

        return word_dictionary, char_dictionary, num_entities

    def parse_one_file(self, fname, dictionary, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname).readlines()
        doc_raw = raw[2].split() # document
        qry_raw = raw[4].split() # query
        ans_raw = raw[6].strip() # answer
        cand_raw = map(lambda x:x.strip().split(':')[0].split(), 
                raw[8:]) # candidate answers

        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)
        try:
            cloze = qry_raw.index('@placeholder')
        except ValueError:
            print '@placeholder not found in ', fname, '. Fixing...'
            at = qry_raw.index('@')
            qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at+2])] + qry_raw[at+2:]
            cloze = qry_raw.index('@placeholder')

        # tokens/entities --> indexes
        doc_words = map(lambda w:w_dict[w], doc_raw)
        qry_words = map(lambda w:w_dict[w], qry_raw)
        if use_chars:
            doc_chars = map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']), 
                list(w)[:MAX_WORD_LEN]), doc_raw)
            qry_chars = map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']), 
                list(w)[:MAX_WORD_LEN]), qry_raw)
        else:
            doc_chars, qry_chars = [], []
        ans = map(lambda w:w_dict.get(w,0), ans_raw.split())
        cand = [map(lambda w:w_dict.get(w,0), c) for c in cand_raw]

        return doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze

    def parse_all_files(self, directory, dictionary, use_chars):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """
        all_files = glob.glob(directory + '/*.question')
        questions = [self.parse_one_file(f, dictionary, use_chars) + (f,) for f in all_files]
        return questions

    def gen_text_for_word2vec(self, question_dir, text_file):

            fnames = []
            fnames += glob.glob(question_dir + "/training/*.question")

            out = open(text_file, "w")

            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline()
                fp.readline()
                query = fp.readline()
                fp.close()
                
                out.write(document.strip())
                out.write(" ")
                out.write(query.strip())

            out.close()

if __name__ == '__main__':
    dp = DataPreprocessor()
    dp.gen_text_for_word2vec("cnn/questions", "/tmp/cnn_questions.txt")

