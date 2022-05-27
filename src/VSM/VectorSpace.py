import numpy as np

def tokenize( s ):
    tokens = list(filter(None, s.split(' ')))
    return tokens

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectorsTF = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]


    def __init__(self, documents={}):
        item_ids = documents.keys()
        
        self.documents = documents
        self.doc_size = len(item_ids)
        self.documentVectorsTF=[]

        
        """ Create the vector space for the passed document strings """
        self.doc_word_list = {}
        for item_id in item_ids:
            self.doc_word_list[item_id] = tokenize(documents[item_id])
        self.vectorKeywordIndex = self.getVectorKeywordIndex()

        self.documentVectorsTF = {}
        for item_id in item_ids:
            self.documentVectorsTF[item_id] = self.makeTFVector(self.doc_word_list[item_id])

        self.makeIDF()
        self.documentVectorsTFIDF = {}
        for item_id in item_ids:
            self.documentVectorsTFIDF[item_id] = self.documentVectorsTF[item_id] * self.idf_v

    # construct idf vector
    def makeIDF(self):
        # count how many doc contain word
        def n_containing(word):
            result = 0
            for word_list in self.doc_word_list.values():
                if word in word_list:
                    result += 1
            return result

        idf_v = np.zeros(self.word_size)

        # calculate idf value for each word
        for word in self.uniqueVocabularyList:
            index = self.vectorKeywordIndex.get(word, -1)
            idf_v[index] = np.log( (self.doc_size / (1 + n_containing(word)) ) )
        self.idf_v = idf_v


    def getVectorKeywordIndex(self):
        """ create the keyword associated to the position of the elements within the document vectors """

        # combine doc_word_list into list of words for the whole corpus
        vocabularyList = []
        for word_list in self.doc_word_list.values():
            vocabularyList.extend(word_list)

        self.uniqueVocabularyList = list(set(vocabularyList))
        self.word_size = len(self.uniqueVocabularyList)
        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in self.uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    # construct tf vector based on document string
    def makeTFVector(self, wordList ):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = np.zeros(self.word_size)
        
        for word in wordList:
            index = self.vectorKeywordIndex.get(word, -1)
            if index!=-1:
                vector[index] += 1; #Use simple Term Count Model
        vector = vector / len(wordList)
        return vector