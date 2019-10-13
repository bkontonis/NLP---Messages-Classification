import re
import sys
import nltk
import joblib
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])




def load_data(database_filepath):
    '''Loads data from database
    
       Input:
           database_filepath: The path and name of the database
       Output:
           X: Message column
           Y: Category dummy columns
           category_names: A list with the category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM LabeledMessages', engine)
    #drop child_alone category that has only zeros. Otherwise SGDClassifier won't be able to run.
    df = df.drop(['child_alone'], axis=1)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.columns[4:]
    return X, Y, category_names


# Create a map between Treebank and WordNet 
# WordNet POS tags are: NOUN = 'n', ADJ = 's', VERB = 'v', ADV = 'r'
# Descriptions (c) https://web.stanford.edu/~jurafsky/slp3/10.pdf
tag_map = {
        'CC':None, # coordin. conjunction (and, but, or)  
        'CD':wn.NOUN, # cardinal number (one, two)             
        'DT':None, # determiner (a, the)                    
        'EX':wn.ADV, # existential ‘there’ (there)           
        'FW':None, # foreign word (mea culpa)             
        'IN':wn.ADV, # preposition/sub-conj (of, in, by)   
        'JJ':wn.ADJ, # adjective (yellow)                  
        'JJR':wn.ADJ, # adj., comparative (bigger)          
        'JJS':wn.ADJ, # adj., superlative (wildest)           
        'LS':None, # list item marker (1, 2, One)          
        'MD':None, # modal (can, should)                    
        'NN':wn.NOUN, # noun, sing. or mass (llama)          
        'NNS':wn.NOUN, # noun, plural (llamas)                  
        'NNP':wn.NOUN, # proper noun, sing. (IBM)              
        'NNPS':wn.NOUN, # proper noun, plural (Carolinas)
        'PDT':wn.ADJ, # predeterminer (all, both)            
        'POS':None, # possessive ending (’s )               
        'PRP':None, # personal pronoun (I, you, he)     
        'PRP$':None, # possessive pronoun (your, one’s)    
        'RB':wn.ADV, # adverb (quickly, never)            
        'RBR':wn.ADV, # adverb, comparative (faster)        
        'RBS':wn.ADV, # adverb, superlative (fastest)     
        'RP':wn.ADJ, # particle (up, off)
        'SYM':None, # symbol (+,%, &)
        'TO':None, # “to” (to)
        'UH':None, # interjection (ah, oops)
        'VB':wn.VERB, # verb base form (eat)
        'VBD':wn.VERB, # verb past tense (ate)
        'VBG':wn.VERB, # verb gerund (eating)
        'VBN':wn.VERB, # verb past participle (eaten)
        'VBP':wn.VERB, # verb non-3sg pres (eat)
        'VBZ':wn.VERB, # verb 3sg pres (eats)
        'WDT':None, # wh-determiner (which, that)
        'WP':None, # wh-pronoun (what, who)
        'WP$':None, # possessive (wh- whose)
        'WRB':None, # wh-adverb (how, where)
        '$':None, #  dollar sign ($)
        '#':None, # pound sign (#)
        '“':None, # left quote (‘ or “)
        '”':None, # right quote (’ or ”)
        '(':None, # left parenthesis ([, (, {, <)
        ')':None, # right parenthesis (], ), }, >)
        ',':None, # comma (,)
        '.':None, # sentence-final punc (. ! ?)
        ':':None # mid-sentence punc (: ; ... – -)
    }

def tokenize(text):     
    ''' Tokenizer for CountVectorizer() 

        Inputs: 
            text: message instance
        Output: 
            clean_tokens: list of lemmatized tokens based on words from the message
    '''
    #remove url links
    re_url = r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
    text = re.sub(re_url, 'urlplaceholder', text)

    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
    tokens = word_tokenize(text)
    
    # remove short words
    tokens = [token for token in tokens if len(token) > 2]
    
    # remove stopwords
    STOPWORDS = list(set(stopwords.words('english')))
    tokens = [token for token in tokens if token not in STOPWORDS]

    pos_tokens = pos_tag(tokens) 

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok, pos in pos_tokens:
        try:
            if tag_map[pos] is not None:
                clean_tok = lemmatizer.lemmatize(tok, tag_map[pos]).lower().strip()
                clean_tokens.append(clean_tok)
            else:
                clean_tok = lemmatizer.lemmatize(tok).lower().strip()
                clean_tokens.append(clean_tok)
        except KeyError:
            pass
            
    return clean_tokens



def build_model():
    '''ML Pipeline to build the appropriate model.

        Input: 
            None
        Output: 
            cv: The model with the best parameters after GridSearchCV for pipeline
                 consisting of nlp steps and final estimator with multioutput wrapper
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(random_state=42)))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Apply the best model on test data and prints some evaluation 
       metrics regarding its performance.
       It prints:
           - Model's best parameters and score
           - The classification report for each category
           - The accuracy score for each category
           - The overall accuracy score for all the categories
       
       Input:
           model: The best model from GridSearchCV
           X_test: Test messages
           Y_test: Test category dummies
           category_names: The list of the category names
       Output: None
    '''
    Y_pred = model.predict(X_test)
    print(model.best_params_, model.best_score_)
    
    for i, col in enumerate(category_names):
        print(i, col)
        print(classification_report(Y_test.to_numpy()[:, i], Y_pred[:, i]))
        print(accuracy_score(Y_test.to_numpy()[:, i], Y_pred[:, i]))
    
    print((Y_test == Y_pred).sum() / Y_test.count())
    print()
    print('Overall Accuracy: {}'.format((Y_test == Y_pred).sum().sum() / Y_test.count().sum()))



def save_model(model, model_filepath):
    '''Save model pickle object 

        Inputs: 
            model: The best model, as output of build_model()
            model_filepath: The filepath you would like to save the model (with subfolders if
            necessary), e.g. "models/model.sav"
        Output: None
    '''
    joblib.dump(model, model_filepath)



def main():
    '''Executes all the steps in the right order
    
       Input: None
       Output: None
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()