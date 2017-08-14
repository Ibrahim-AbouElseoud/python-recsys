=============
python-recsys
=============

A python library for implementing a recommender system.

Incremental SVD update for python-recsys
========================================
- python-recsys now supports incrementally adding new users or items instead of building the model from scratch for these new users or items via the folding-in technique which was mentioned in Sarwar et al.'s `paper`_ (Titled: Incremental Singular Value Decomposition Algorithms for Highly Scalable Recommender Systems), this latest commit is simply an implementation to it for python-recsys.

.. _`paper`: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.7894&rep=rep1&type=pdf

- A `Demonstration video is available`_  for this latest commit in form of a demo site built using the MEAN stack which uses the updated python-recsys as backend for the recommender which folds-in the website's user in to the SVD model and gets recommendations instantaneously instead of building the model from scratch.

.. _`Demonstration video is available`:  https://youtu.be/tIvQxBfa2d4

- There is also an accompanying `bachelor thesis paper`_ (For those interested) which outlines the background, architecture and discusses the "Folding-in" approach.

.. _`bachelor thesis paper`: https://drive.google.com/file/d/0BylQe2cRVWE_RmZoUTJYSGZNaXM/view

Installation
============

Dependencies
~~~~~~~~~~~~

**python-recsys** is build on top of `Divisi2`_, with csc-pysparse (Divisi2 also requires `NumPy`_, and uses Networkx).

.. _`Divisi2`: http://csc.media.mit.edu/docs/divisi2/install.html
.. _`NumPy`: http://numpy.scipy.org

**python-recsys** also requires `SciPy`_.

.. _`SciPy`: http://numpy.scipy.org

To install the dependencies do something like this (Ubuntu):

::

    sudo apt-get install python-scipy python-numpy
    sudo apt-get install python-pip
    sudo pip install csc-pysparse networkx divisi2

    # If you don't have pip installed then do:
    # sudo easy_install csc-pysparse
    # sudo easy_install networkx
    # sudo easy_install divisi2

Download
~~~~~~~~

Download **python-recsys**  from `github`_.

.. _`github`: http://github.com/ocelma/python-recsys

Install
~~~~~~~

::

    tar xvfz python-recsys.tar.gz
    cd python-recsys
    sudo python setup.py install

Example
~~~~~~~

1. Load Movielens dataset:

::

    from recsys.algorithm.factorize import SVD
    svd = SVD()
    svd.load_data(filename='./data/movielens/ratings.dat',
                sep='::',
                format={'col':0, 'row':1, 'value':2, 'ids': int})

2. Compute Singular Value Decomposition (SVD), M=U Sigma V^t:

::

    k = 100
    svd.compute(k=k,
                min_values=10,
                pre_normalize=None,
                mean_center=True,
                post_normalize=True,
                savefile='/tmp/movielens')

3. Get similarity between two movies:

::

    ITEMID1 = 1    # Toy Story (1995)
    ITEMID2 = 2355 # A bug's life (1998)

    svd.similarity(ITEMID1, ITEMID2)
    # 0.67706936677315799

4. Get movies similar to *Toy Story*:

::

    svd.similar(ITEMID1)

    # Returns: <ITEMID, Cosine Similarity Value>
    [(1,    0.99999999999999978), # Toy Story
     (3114, 0.87060391051018071), # Toy Story 2
     (2355, 0.67706936677315799), # A bug's life
     (588,  0.5807351496754426),  # Aladdin
     (595,  0.46031829709743477), # Beauty and the Beast
     (1907, 0.44589398718134365), # Mulan
     (364,  0.42908159895574161), # The Lion King
     (2081, 0.42566581277820803), # The Little Mermaid
     (3396, 0.42474056361935913), # The Muppet Movie
     (2761, 0.40439361857585354)] # The Iron Giant

5. Predict the rating a user (USERID) would give to a movie (ITEMID):

::

    MIN_RATING = 0.0
    MAX_RATING = 5.0
    ITEMID = 1
    USERID = 1

    svd.predict(ITEMID, USERID, MIN_RATING, MAX_RATING)
    # Predicted value 5.0

    svd.get_matrix().value(ITEMID, USERID)
    # Real value 5.0

6. Recommend (non-rated) movies to a user:

::

    svd.recommend(USERID, is_row=False) #cols are users and rows are items, thus we set is_row=False

    # Returns: <ITEMID, Predicted Rating>
    [(2905, 5.2133848204673416), # Shaggy D.A., The
     (318,  5.2052108435956033), # Shawshank Redemption, The
     (2019, 5.1037438278755474), # Seven Samurai (The Magnificent Seven)
     (1178, 5.0962756861447023), # Paths of Glory (1957)
     (904,  5.0771405690055724), # Rear Window (1954)
     (1250, 5.0744156653222436), # Bridge on the River Kwai, The
     (858,  5.0650911066862907), # Godfather, The
     (922,  5.0605327279819408), # Sunset Blvd.
     (1198, 5.0554543765500419), # Raiders of the Lost Ark
     (1148, 5.0548789542105332)] # Wrong Trousers, The

7. Which users should *see* Toy Story? (e.g. which users -that have not rated Toy
   Story- would give it a high rating?)

::

    svd.recommend(ITEMID)

    # Returns: <USERID, Predicted Rating>
    [(283,  5.716264440514446),
     (3604, 5.6471765418323141),
     (5056, 5.6218800339214496),
     (446,  5.5707524860615738),
     (3902, 5.5494529168484652),
     (4634, 5.51643364021289),
     (3324, 5.5138903299082802),
     (4801, 5.4947999354188548),
     (1131, 5.4941438045650068),
     (2339, 5.4916048051511659)]

Example for incremental update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load Movielens dataset and prepare for training and testing:

::

    import recsys.algorithm
    recsys.algorithm.VERBOSE = True

    from recsys.algorithm.factorize import SVD
    from recsys.datamodel.data import Data

    filename = “(your movielens file path here)”

    #In movielens dataset, the user is at 0 so I made them the row (could keep it as above {'col':0, 'row':1, 'value':2, 'ids': int} but I changed order to emphasis a parameter in an upcoming function)
    format = {'col':1, 'row':0, 'value':2, 'ids': int}

    data = Data()
    data.load(filename, sep='::', format=format)
    #splits the dataset according to row or column (based on is_row=true or false) which causes there to be no overlap (of users for example) between train and foldin dataset
    train, test, foldin = data.split_train_test_foldin(base=60,percentage_base_user=80,shuffle_data=True,is_row=True) #since users are in the row so is_row=true

    # Returns: a tuple <Data, Data, Data> for train, test, foldin
    # Prints: (If VERBOSE=True)
    total number of tuples: 1000209
    percentage of data for training: 48.0 % with 479594 tuples
    percentage of data for testing: 20.0 % with 200016 tuples # 100-percentage_base_user per user (percentage of tuples which means the ratings since a user has many tuples(ratings))
    percentage of data for foldin: 32.0 % with 320599 tuples
    _____________
    percentage of users for foldin: 40.0 % with 2416 users # 100-base= foldin (percentage of users)
    percentage of users for training: 60.0 % with 3624 users #base for training (percentage of users)

2. Compute Singular Value Decomposition (SVD), M=U Sigma V^t:

::

    svd = SVD()
    svd.set_data(train)
    svd.compute(k=100,
            	min_values=1,
            	pre_normalize=None,
            	mean_center=False,
            	post_normalize=True)

    # Prints:
    Creating matrix (479594 tuples)
    Matrix density is: 3.7007%
    Updating matrix: squish to at least 1 values
    Computing svd k=14, min_values=1, pre_normalize=None, mean_center=False, post_normalize=False

3. "Foldin" those new users or items (update model instead of updating from scratch)

::

    svd.load_updateDataBatch_foldin(data=foldin,is_row=True)

    # Prints: (If VERBOSE=True)
    before updating, M= (3624, 3576)
    done updating, M= (6040, 3576) # Folds in all the new users (not previously in model)

4. Recommend (non-rated) movies to a NEW user
::

    user_id=foldin[0][1] #returns userID which is in foldin dataset BUT not in train dataset
    svd.recommend(user_id,is_row=True,only_unknowns=True) #The userID is in row and gets only the unrated (unknowns)

    # Returns: <ITEMID, Predicted Rating>
    [(1307, 3.6290483094468913),
    (1394, 3.5741565545425957),
    (1259, 3.5303836262378048),
    (1968, 3.4565426585553927),
    (2791, 3.3470277643217203),
    (1079, 3.268283171487782),
    (1198, 3.2381080336246675),
    (593, 3.204915630088236),
    (1270, 3.1859618303393233),
    (2918, 3.1548530640630252)]

5. Recommend (non-rated) movies to a NEW user and validate not in base model (prior to folding-in)
::

    # BEFORE running points 3 and 4 (prior to calling svd.load_updateDataBatch_foldin)

    user_id=foldin[0][1] #returns userID which is in foldin dataset BUT not in train dataset

    # Try block to validate that the userID is new and not in the base model
    try:
        print "Getting recommendation for user_id which was not in original model training set"
        print "recommendations:",svd.recommend(user_id)
    except Exception:
        print "New user not in base model so in except block and will foldin the foldin dataset (update the model NOT calculate from scratch)"
        svd.load_updateDataBatch_foldin(data=foldin,format=format,is_row=True,truncate=True,post_normalize=True)
        print "recommendations:",svd.recommend(user_id,is_row=True,only_unknowns=True) #The userID is in row and get us only the unrated (unknowns)


    # Prints:
    Getting recommendation for user_id which was not in original model training set
    recommendations: New user not in base model so in except block and will foldin the foldin dataset (update the model NOT calculate from scratch)
    before updating, M= (3624, 3576)
    done updating, M= (6040, 3576)
    recommendations: [(1307, 3.6290483094468913), (1394, 3.5741565545425957), (1259, 3.5303836262378048), (1968, 3.4565426585553927), (2791, 3.3470277643217203), (1079, 3.268283171487782), (1198, 3.2381080336246675), (593, 3.204915630088236), (1270, 3.1859618303393233), (2918, 3.1548530640630252)]


6. Load previous SVD model and foldin NEW users from file then instantly get recommendations
::

    format = {'col':1, 'row':0, 'value':2, 'ids': int}

    svd = SVD()
    #load base svd model
    svd.load_model('SVDModel')

    # load new users by their movie rating data file and use it to fold-in the users into the model (loads data and folds in)
    svd.load_updateDataBatch_foldin(filename = 'newUsers.dat', sep='::', format=formate, is_row=True)

    # gets recommendedations
    print "recommendations:", svd.recommend(new_userID,is_row=True,only_unknowns=True)


- All the normal functionalities of python-recsys are compatible with the incremental update commit. The incremental update can even work if you load the model then foldin a new user or users or even items.

- Please note that preexisting users can't be folded-in only new users which aren't already in the svd model.

Documentation
~~~~~~~~~~~~~

Documentation and examples available `here`_.

.. _`here`: http://ocelma.net/software/python-recsys/build/html

To create the HTML documentation files from doc/source do:

::

    cd doc
    make html

HTML files are created here:

::

    doc/build/html/index.html
