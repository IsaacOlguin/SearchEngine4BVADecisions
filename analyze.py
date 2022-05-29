#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
###General
import sys
import os.path as Path
from datetime import datetime
###Specific
import multiprocessing as mp
import spacy
import numpy as np
from joblib import load
import fasttext
from tabulate import tabulate
from art import text2art
###Custom
import luima_sbd.sbd_utils as sbd_utils
import config.project_config as project_config
import config.sentences_custom as sentences_custom

#########################################################################
#########################################################################

LOG_LEVEL = 2
LOG_NOTHING = 0
LOG_ERROR = 1
LOG_INFO = 2
LOG_WARN=3
LOG_DEBUG = 4

def infoLog(message):
    if LOG_LEVEL >= LOG_INFO:
        now = datetime.now()
        print(now.strftime("[INFO ] %H:%M:%S"), message)

def debugLog(message):
    if LOG_LEVEL >= LOG_DEBUG:
        now = datetime.now()
        print(now.strftime("[DEBUG] %H:%M:%S"), message)

def errorLog(message):
    if LOG_LEVEL >= LOG_ERROR:
        now = datetime.now()
        print(now.strftime("\n\n\t\t[ERROR] %H:%M:%S"), message)

#########################################################################
#########################################################################

def main():
    if 1 == len(sys.argv) or len(sys.argv) > 2:
        infoLog("The execution of this file requires only one parameter (which is the name/path of the file to analyse)")
        return 

    pathfile = sys.argv[1]

    if not Path.isfile(pathfile):
        infoLog(f"The file does not exist. Please confirm the location of your file <<{pathfile}>>")
        return

    infoLog(f"Reading file ({pathfile})...")
    content_file = ""
    with open(pathfile, 'r', encoding="latin1") as f:
        content_file = f.read()
    f.close()

    #print(content_file)
    infoLog("Extracting sentences...")
    sentences = [sent for sent in sbd_utils.text2sentences(content_file)]
    debugLog(f'Number of sentences in the file according to the law-specific sentence segmenter {len(sentences)}')

    list_sentences = [{"txt":t, "start_normalized":(content_file.index(t)/len(content_file))} for t in sentences]
    #for t in list_sentences: debugLog(t)

    infoLog("Reading embeddings...")
    path_current_file = Path.abspath('')
    pathModel = Path.join(path_current_file, project_config.PATH_MODEL_FASTTEXT)
    modelFastText = fasttext.load_model(pathModel)
    infoLog(f'Path model: {pathModel}')
    #debugLog(f'\n{modelFastText.get_word_vector("the")}')

    mean, std_dev, listNumTokensNorm, listTokens = getMeanAndStdDevTokensTraining(list_sentences)

    debugLog(f'Mean={mean}')
    debugLog(f'StdDev={std_dev}')
    debugLog(f'ListNumTokensNorm={len(listNumTokensNorm)} => ')#{listNumTokensNorm}
    debugLog(f'listTokens={len(listTokens)}')
    debugLog(list_sentences[0])

    vectorized = [getVectorFeature(t, modelFastText) for t in listTokens]

    debugLog(f'Len received vector {len(vectorized)},  with shape {vectorized[0].shape}, and type {type(vectorized[0])}')

    vector_np = np.asarray(vectorized, dtype='float32')
    starts_normalized = np.array([s['start_normalized'] for s in list_sentences])
    tokens_normalized = np.array([s['tokens_normalized'] for s in list_sentences])

    debugLog(f'Shape of vector_np before concatenation={vector_np.shape}')

    X = np.concatenate((vector_np, np.expand_dims(starts_normalized, axis=1)), axis=1)
    X = np.concatenate((X, np.expand_dims(tokens_normalized, axis=1)), axis=1)

    debugLog(f'Shape of X after concatenation={X.shape}')
    
    ##########################################################################
    infoLog("Importing word embeddings...")
    pathWordEmbeddingNonLinear = Path.join(path_current_file, project_config.PATH_WORD_EMBEDDING_NON_LINEAR)
    infoLog(f'Path word embeddings non-linear: {pathWordEmbeddingNonLinear}')
    clf = load(pathWordEmbeddingNonLinear)
    debugLog("Model has been imported")

    prediction = clf.predict(X)

    print(text2art("SVM - RFB"))

    try:
        print(tabulate({"Prediction (Class)": prediction, "Sentences": sentences}, headers='keys'))#, tablefmt='fancy_grid'
    except Error as error:
        errorLog("Error when printing table with tabulate. Was it installed correctly?")
        print(error)

        infoLog("="*80)
        for class_predicted, text in zip(prediction, sentences):
            print(class_predicted, "\t=>\t", text.replace("\r", "").replace("\n", ""))
        infoLog("="*80)

    del X
    del prediction
    del clf
    del vector_np
    del starts_normalized
    del tokens_normalized
    del vectorized
    del modelFastText
    del mean
    del std_dev
    del listNumTokensNorm
    del listTokens
    del content_file
    del pathfile


#########################################################################
#########################################################################

def getTxtFromSpan(span):
    return span["txt"].lower()

def getMeanAndStdDevTokensTraining(spans):
    debugLog("BEGIN getMeanAndStdDevTokensTraining")    
    
    mean = 0.0
    std_dev = 0.0
    nlp = sentences_custom.getNlpObj()
    
    #listNumTokens = [len(sentences_custom.get_tokens_spacy(t["txt"], nlp)) for t in train_spans]
    pool = mp.Pool(mp.cpu_count())
    debugLog("To execute map getTxtFromSpan")
    spans_txt = pool.map(getTxtFromSpan, spans)
    
    debugLog("To execute pipe to get SpacyDocuments")
    #spacy.prefer_gpu()
    nlp = sentences_custom.getNlpObj()
    spacy_docs_from_spans = list(nlp.pipe(spans_txt))
    
    debugLog(f'Length of spacy_docs_from_spans is {len(spacy_docs_from_spans)}')
    
    debugLog("To execute map get_tokens_spacy_opt")
    #listNumTokens = pool.starmap(sentences_custom.get_tokens_spacy_opt, zip(spacy_docs_from_spans, spans_txt))
    listTokens = pool.map(sentences_custom.get_tokens_spacy, spacy_docs_from_spans)
    
    debugLog("Converting npListNumTokens, computing mean, std_dev")
    npListNumTokens = [len(t) for t in listTokens]
    npListNumTokens = np.asarray(npListNumTokens, dtype='float32')
    
    mean = npListNumTokens.mean()
    std_dev = npListNumTokens.std()
    
    listNumTokensNorm = (npListNumTokens-mean)/std_dev
    
    debugLog("Adding tokens_normalized to each span")    
    for span, tokens_norm in zip(spans, listNumTokensNorm):
        span["tokens_normalized"] = tokens_norm
    
    debugLog("END getMeanAndStdDevTokensTraining")    
    return mean, std_dev, listNumTokensNorm, listTokens

def getVectorFeature(tokens, modelFastText):
    #print("Num tokens =", len(tokens))
    #print(tokens)

    vectors = [modelFastText.get_word_vector(t) for t in tokens]
    
    vector = np.asarray(vectors, dtype='float32').mean(axis=0)
        
    return vector
    
#########################################################################
#########################################################################

if __name__ == "__main__":
    main()

#########################################################################
#########################################################################

