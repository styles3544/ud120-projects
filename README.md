ud120-projects (edited by: Piyush Kumar - styles)
==============

Starter project code for students taking Udacity ud120

### Contribution Note
The code in this repo has been upgraded from Python 2 to Python 3 by [Siddharth Kekre](https://github.com/iSiddharth20) in this [PR](https://github.com/udacity/ud120-projects/pull/302). Refer to the [Change Log](https://github.com/iSiddharth20/ud120-projects/blob/master/CHANGELOG.md) for more details. 

### Fixing Errors 
Refer to the link [click here](https://github.com/udacity/ud120-projects/issues/181#issuecomment-895850262)

=========================================================================
##### For users using python3 (python 3.8) instead of python2. Follow this steps to fix the issue.

1. Create a file named "doc2unix.py" in the directory /tools/  ---> This is the content of the file (utilized @hat20 answer)

```
#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: python dos2unix.py
"""

import sys

original = 'word_data.pkl'
destination = "word_data_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))
```

2. Run this file. Make sure there are no errors.
3. The email_preprocess file has certain errors related to python libraries. CHanged a couple of lines in the email_preprocess file to resolve the errors:

```
#!/usr/bin/python

import pickle
#import cPickle
import numpy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



def preprocess(words_file = "../tools/word_data_unix.pkl", authors_file="../tools/email_authors.pkl"):
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "rb")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "rb")
    word_data = pickle.load(words_file_handler)
    words_file_handler.close()

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)



    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print("no. of Chris training emails:", sum(labels_train))
    print("no. of Sara training emails:", len(labels_train)-sum(labels_train))

    return features_train_transformed, features_test_transformed, labels_train, labels_test
    
    
    
    
