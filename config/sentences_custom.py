import os
import multiprocessing as mp
import config.project_config as project_config
import luima_sbd.sbd_utils as sbd_utils
import chardet

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.symbols import ORTH

def getNlpObj():
    #spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")
    """
    nlp.tokenizer.add_special_case('1.', [{ORTH: '1.'}])
    nlp.tokenizer.add_special_case('2.', [{ORTH: '2.'}])
    nlp.tokenizer.add_special_case('3.', [{ORTH: '3.'}])
    nlp.tokenizer.add_special_case('a.', [{ORTH: 'a.'}])
    nlp.tokenizer.add_special_case('b.', [{ORTH: 'b.'}])
    nlp.tokenizer.add_special_case('c.', [{ORTH: 'c.'}])
    nlp.tokenizer.add_special_case('I.', [{ORTH: 'I.'}])
    nlp.tokenizer.add_special_case('II.', [{ORTH: 'II.'}])
    nlp.tokenizer.add_special_case('III.', [{ORTH: 'III.'}])
    nlp.tokenizer.add_special_case('REPRESENTATION', [{ORTH: 'REPRESENTATION'}])
    nlp.tokenizer.add_special_case('THE ISSUE', [{ORTH: 'THE ISSUE'}])
    nlp.tokenizer.add_special_case('ATTORNEY FOR THE BOARD', [{ORTH: 'ATTORNEY FOR THE BOARD'}])
    nlp.tokenizer.add_special_case('ORDER', [{ORTH: 'ORDER'}])
    nlp.tokenizer.add_special_case('FINDINGS OF FACT', [{ORTH: 'FINDINGS OF FACT'}])
    nlp.tokenizer.add_special_case('CONCLUSION OF LAW', [{ORTH: 'CONCLUSION OF LAW'}])
    nlp.tokenizer.add_special_case('INTRODUCTION', [{ORTH: 'INTRODUCTION'}])
    nlp.tokenizer.add_special_case('WITNESS AT HEARINGS ON APPEAL', [{ORTH: 'WITNESS AT HEARINGS ON APPEAL'}])
    nlp.tokenizer.add_special_case('REASONS AND BASES FOR FINDINGS AND CONCLUSION', [{ORTH: 'REASONS AND BASES FOR FINDINGS AND CONCLUSION'}])
    nlp.tokenizer.add_special_case('THE ISSUES', [{ORTH: 'THE ISSUES'}])
    """

    return nlp

"""
def spacy_tokenize(txt):
    nlp = getNlpObj()
    doc = nlp(txt)
    tokens = list(doc)
    clean_tokens = []
    for t in tokens:
        #print(f'{t.text} | {t.lemma_} | {t.pos_}')
        if t.pos_ == 'PUNCT' or t.pos_ == 'SPACE': #Removes all punctuation
            pass
        elif t.pos_ == 'NUM':
            clean_tokens.append(f'<NUM{len(t)}>')
        else:
            clean_tokens.append(t.lemma_)
    return clean_tokens
"""
"""
def spacy_tokenize(doc):
    tokens = list(doc)
    clean_tokens = []

    for t in tokens:
        if t.pos_ == 'PUNCT' or t.pos_ == 'SPACE': #Removes all punctuation
            pass
        elif t.pos_ == 'NUM':
            clean_tokens.append(f'<NUM{len(t)}>')
        elif t.pos_ == 'PROPN' and t.text == '�':
            print(f'{t.text} | {t.lemma_} | {t.pos_}')
            pass
        else:
            print(f'{t.text} | {t.lemma_} | {t.pos_}')
            clean_tokens.append((str(t.lemma_)).lower())
    return clean_tokens
"""

def get_tokens_spacy(doc, nlp=None):
    #spacy.prefer_gpu()
    if nlp==None: 
        #print("crear nlp")
        nlp = getNlpObj() 
        if(type(doc) == 'string'):
            doc = nlp(doc.lower())
    if(type(doc) == 'string'):
        doc = nlp(doc.lower()) #Check if its already an object can be removed
    tokens = list(doc)
    #list_stop_words = nlp.Defaults.stop_words
    clean_tokens = []

    for t in tokens:
        #print(f'{t.text} | {t.lemma_} | {t.pos_}')
        if t.pos_ == 'PUNCT' or t.pos_ == 'SPACE': #Removes all punctuation
            pass
        elif t.pos_ == 'NUM':
            clean_tokens.append(f'<NUM{len(t)}>')
        elif t.pos_ == 'PROPN' and t.text == '�':
            #print(f'{t.text} | {t.lemma_} | {t.pos_}')
            pass
        else:
            #print(f'{t.text} | {t.lemma_} | {t.pos_}')
            clean_tokens.append(t.lemma_)

    return clean_tokens#[t for t in clean_tokens if t not in list_stop_words]

def getLengthSententesWithSegmenterPerFile(filename):
    print(filename, end = '')
    path_file = os.path.join(project_config.UNLABELED_DIR, filename)
    #print(f'{mp.current_process()} - {filename} - {path_file}\n')  
    #print(filename, end="")  
    try:         
        raw = open(path_file, 'rb').read()
        enc = chardet.detect(raw)['encoding']
        #with codecs.open(path_file, mode='r', encoding=enc) as f:
        with open(path_file, encoding='latin-1') as f:
            return len(sbd_utils.text2sentences(f.read()))
    except Error as e:
        print(e)
        return 0
    
    return 0


def getSententesWithSegmenterPerFile(filename):
    print(filename, end = '')
    path_file = os.path.join(project_config.UNLABELED_DIR, filename)
    #print(f'{mp.current_process()} - {filename} - {path_file}\n')  
    #print(filename, end="") 
    
    try:         
        raw = open(path_file, 'rb').read()
        enc = chardet.detect(raw)['encoding']
        #with codecs.open(path_file, mode='r', encoding=enc) as f:
        with open(path_file, encoding='latin-1') as f:
            sentences = sbd_utils.text2sentences(f.read())
            resultado = {"name":filename, "num_sentences":len(sentences), "sentences": sentences}
            return resultado
    except Error as e:
        print(e)
        return {"name":filename, "num_sentences":0, "sentences": []}
    
    return {"name":filename, "num_sentences":0, "sentences": []}


def getNumTokensPerFile(obj):

    namefile_token = "output/o_" + obj.get("name")
    list_files_tokens = os.listdir(os.path.join(project_config.OUTPUT_DIR))
    if obj.get("name") in list_files_tokens:
        return 0
    print("N", end="")

    obj['num_tokens_spacy_per_sentence'] = []
    nlp = getNlpObj()
    #nlp.disable_pipes('parser')
    #nlp.pipe(texts, n_process=4)
    #print("Por procesar sentences")
    docSentences = list(nlp.pipe(obj.get("sentences"), n_process=1))
    #print("Termine de procesar sentences")
    #print(mp.cpu_count())

    file = open(namefile_token, "w")
    for sentence in docSentences:
        tokens = get_tokens_spacy(sentence, nlp)
        obj['num_tokens_spacy_per_sentence'].append(len(tokens))

        if len(tokens) > 5:
            tokenString = " ".join(tokens)
            tokenString += "\n"
            file.write(tokenString)
    file.close()
    return sum(obj["num_tokens_spacy_per_sentence"])

def getListNumTokensPerFile(obj):
    obj['num_tokens_spacy_per_sentence'] = []
    nlp = getNlpObj()
    docSentences = list(nlp.pipe(obj.get("sentences"), n_process=1))

    for sentence in docSentences:
        tokens = get_tokens_spacy(sentence, nlp)
        obj['num_tokens_spacy_per_sentence'].append(len(tokens))
        
    return sum(obj["num_tokens_spacy_per_sentence"])
