import pandas as pd
import numpy as np
import sys

import tldextract
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

import editdistance

from tqdm import tqdm
import bs4
import requests

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric

from tqdm.notebook import tqdm

class FeatureEmbeddings:
    def __init__(self):
        self.features = pd.DataFrame()
        n = 2
        self.bigram = MLE(n)

    def __URLsplit(self,s):
        return [char for char in s]

    def __buildBigram(self,urls):
        train_data, padded_sents = padded_everygram_pipeline(2, urls)
        self.bigram.fit(train_data,padded_sents)

    def __cleanURL(self,url):
        xtract = tldextract.extract(url)
        return '.'.join(xtract)

    def __editDistance(self,url):
        popular_sites = ['https://news.yahoo.com/','https://news.google.com/?hl=en-US&gl=US&ceid=US:en',
                        'https://www.huffpost.com/','https://www.cnn.com/','https://www.nytimes.com/',
                        'https://www.foxnews.com/','https://www.nbcnews.com/',
                        'https://www.dailymail.co.uk/ushome/index.html','https://www.washingtonpost.com/',
                        'https://www.theguardian.com/us','https://www.wsj.com/','https://abcnews.go.com/',
                        'https://www.bbc.co.uk/news','https://www.usatoday.com/',
                        'https://www.latimes.com/']
        popular_sites = [self.__cleanURL(str(x)) for x in popular_sites]
        dist = float('inf')
        for site in popular_sites:
            new_dist = editdistance.eval(url,site)
            if new_dist < dist:
                dist = new_dist
        return dist

    def __htmlInfo(self,urls):
        n = len(urls)
        status_codes = [-1]*n
        is_active = [0]*n
        has_wp_content = [-1]*n
        num_iframes = [-1]*n
        it = -1
        for url in tqdm(urls):
            it += 1
            try:
                response = requests.get(url, timeout=10)
                status_codes[it] = response.status_code

                if response.status_code == 200:
                    page = bs4.BeautifulSoup(response.text, 'lxml')
                    is_active[it] = 1
                    iframes = page.find_all(name='iframe')
                    num_iframes[it] = len(iframes)
                    has_wp_content[it] = 1 if response.text.find('wp-content') > -1 else 0
            except:
                continue
        self.features['status'] = status_codes
        self.features['active'] = is_active
        self.features['wp_content'] = has_wp_content
        self.features['num_iframes'] = num_iframes

    def __cleanHeadline(self,h):
        return remove_stopwords(strip_punctuation(strip_numeric(str(h).lower()))).split(' ')

    def __get_val(self,v,row,i):
        if v[row] == []:
            return 0.0
        else:
            return float(v[row][i])

    def __headerEmbeddings(self,headers):
        header_model = Word2Vec.load("models/headline_word_embeddings.model")
        head_vecs = []
        for h in headers:
            h = self.__cleanHeadline(h)
            h = [x for x in h if x in header_model.wv.vocab]
            if len(h) >= 1:
                head_vecs.append(np.mean(header_model[h],axis=0))
            else:
                head_vecs.append([])
        for i in range(len(head_vecs[0])):
            self.features.insert(i,'h_vec_'+str(i),[self.__get_val(head_vecs,row,i) for row in range(len(head_vecs))],True)

    def __articleEmbeddings(self,articles):
        doc_model = Doc2Vec(vector_size=100, window=10, min_count=2, epochs=100)
        doc_model = Doc2Vec.load("models/my_doc2vec_model")
        a_vec_labels = []
        for i in range(0,100):
            a_vec_labels.append('a_vec_'+str(i))
        vecs = []
        loop = tqdm(total=len(articles), position=0) # the little progress bar thing
        for i, text in enumerate(articles):
            loop.set_description('Inferring vector for article number: '+str(i))
            loop.update(1)
            t = text.split()
            e = list(doc_model.infer_vector(t))
            vecs.append(e)
        a_embeds = pd.DataFrame(vecs,columns=a_vec_labels)
        self.features = a_embeds.join(self.features)

    def create(self,data,url_col,article_col,header_col=None):
        '''
        Creates feature dataset from news article URL
        Features:
          BUILT:
            TRANSFERRED:
              - bigram entropy
              - bigram perplexity
              - clean bigram entropy
              - clean bigram perplexity
              - edit distance to top 15 site
              - status
              - active
              - has wordpress content
              - number of iframes
            NEW:
              - header embeddings
          TO BE BUILT:
            - article embeddings
            - url embeddings
        '''
        # HEADLINE VECTORS
        if header_col:
            sys.stdout.write('Building embeddings for headlines...\n')
            self.__headerEmbeddings(data[header_col])

        if url_col is not None:
            # BIGRAM ENTROPY & PERPLEXITY
            sys.stdout.write('Building bigram model features for URL strings...\n')
            urls = data[url_col].apply(lambda a: str(a))
            split_urls = urls.apply(lambda a: self.__URLsplit(a))
            self.__buildBigram(split_urls)
            self.features['bigram_entropy'] = [self.bigram.entropy(x) for x in urls]
            self.features['bigram_perplexity'] = [self.bigram.perplexity(x) for x in urls]

            # CLEAN BIGRAM ENTROPY & PERPLEXITY
            clean_urls = urls.apply(lambda a: self.__cleanURL(str(a)))
            split_clean_urls = clean_urls.apply(lambda a: self.__URLsplit(a))
            self.__buildBigram(split_clean_urls)
            self.features['clean_bigram_entropy'] = [self.bigram.entropy(x) for x in split_clean_urls]
            self.features['clean_bigram_perplexity'] = [self.bigram.perplexity(x) for x in split_clean_urls]

            # EDIT DISTANCE
            sys.stdout.write('Calculating edit distance for each URL string...\n')
            self.features['edit_distance'] = [self.__editDistance(x) for x in clean_urls]

            # HTML INFO (STATUS, ACTIVE, WP CONTENT, # IFRAMES)
            #sys.stdout.write('Accessing request info for features...\n')
            #self.__htmlInfo(urls)
        
        # ARTICLE EMBEDDINGS VIA DOC2VEC
        sys.stdout.write('Inferring article embeddings via doc2vec...\n')
        self.__articleEmbeddings(data[article_col])
        sys.stdout.flush()

