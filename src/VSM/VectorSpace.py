from pprint import pprint
from Parser import Parser
import util
import numpy as np

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectorsTF = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents=[], tf_only=False, chinese=False):
        self.documents = documents
        self.doc_size = len(self.documents)
        self.documentVectorsTF=[]
        self.parser = Parser()
        self.chinese = chinese

        # set stopwords source
        self.stopwords = set()

        
        """ Create the vector space for the passed document strings """
        self.doc_word_list = [self.str2words(doc) for doc in documents]
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectorsTF = [self.makeTFVector(document) for document in documents]

        if not tf_only:
            self.makeIDF()
            self.documentVectorsTFIDF = [tf_v * self.idf_v for tf_v in self.documentVectorsTF]

    # construct idf vector
    def makeIDF(self):
        # count how many doc contain word
        def n_containing(word):
            result = 0
            for word_list in self.doc_word_list:
                if word in word_list:
                    result += 1
            return result

        idf_v = np.zeros(self.word_size)

        # calculate idf value for each word
        for word in self.uniqueVocabularyList:
            index = self.vectorKeywordIndex.get(word, -1)
            idf_v[index] = np.log( (self.doc_size / (1 + n_containing(word)) ) )
        self.idf_v = idf_v


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        # combine doc_word_list into list of words for the whole corpus
        vocabularyList = []
        for word_list in self.doc_word_list:
            vocabularyList += word_list

        self.uniqueVocabularyList = util.removeDuplicates(vocabularyList)
        self.word_size = len(self.uniqueVocabularyList)
        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in self.uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    # construct tf vector based on document string
    def makeTFVector(self, wordString ):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = np.zeros(self.word_size)

        wordList = self.str2words(wordString)
        
        for word in wordList:
            index = self.vectorKeywordIndex.get(word, -1)
            if index!=-1:
                vector[index] += 1; #Use simple Term Count Model
        vector = vector / len(wordList)
        return vector

    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeTFVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectorsTF[documentId], documentVector) for documentVector in self.documentVectorsTF]
        #ratings.sort(reverse=True)
        return ratings


    def search(self,searchList, method = 1, feedback = False):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        # use different vectors and metrics for different methods
        if method == 1:
            vectors = self.documentVectorsTF
            similarity_func = util.cosine
        elif method == 2:
            vectors = self.documentVectorsTF
            similarity_func = util.euclidean
        elif method == 3:
            vectors = self.documentVectorsTFIDF
            similarity_func = util.cosine
        elif method == 4:
            vectors = self.documentVectorsTFIDF
            similarity_func = util.euclidean

        # get rating for each doc
        ratings = [similarity_func(queryVector, documentVector) for documentVector in vectors]

        # pseudo feedback
        if feedback:
            # find best result in previous method
            best_result_str = self.documents[np.argmax(ratings)]

            best_result_words = self.str2words(best_result_str)
            best_result_words = util.filterNnV(best_result_words)
            
            combineQueryVector = queryVector + 0.5 * self.buildQueryVector(best_result_words)
            ratings = [similarity_func(combineQueryVector, documentVector) for documentVector in vectors]

        #ratings.sort(reverse=True)
        return ratings
    
    # convert string into a list of processed words
    def str2words(self, doc_str, pos = False):
        # temp fix
        words = doc_str.split(') (')
        for i in range(len(words)):
            if words[i].startswith('('):
                words[i] = words[i][1:]
            if words[i].endswith(')'):
                words[i] = words[i][:-1]
        # print(words)
        return words 

    # print out idf vector of the query. for test only.
    def print_query_idf( self, query_list):
        query_list = self.str2words(' '.join(query_list))
        for word in query_list:
            index = self.vectorKeywordIndex.get(word, -1)
            print(word, self.idf_v[index] )