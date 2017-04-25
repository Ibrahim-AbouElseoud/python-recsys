# -*- coding: utf-8 -*-
"""
.. module:: algorithm
   :synopsis: Factorization recsys algorithms

.. moduleauthor:: Oscar Celma <ocelma@bmat.com>

"""
import os
import sys
import zipfile

try:
    import divisi2
except:
    from csc import divisi2
from numpy import loads, mean, sum, nan
from operator import itemgetter

from scipy.cluster.vq import kmeans2 #for kmeans method
from random import randint #for kmeans++ (_kinit method)
from scipy.linalg import norm #for kmeans++ (_kinit method)
from scipy import array #for kmeans method

from numpy import fromfile #for large files (U and V)
from divisi2 import DenseVector
from divisi2 import DenseMatrix
from divisi2.ordered_set import OrderedSet
                                        
from recsys.algorithm.baseclass import Algorithm
from recsys.algorithm.matrix import SimilarityMatrix
from recsys.algorithm import VERBOSE

from numpy.linalg import inv #for update
import numpy as np
from recsys.datamodel.data import Data

TMPDIR = '/tmp'

class SVD(Algorithm):
    """
    Inherits from base class Algorithm. 
    It computes SVD (Singular Value Decomposition) on a matrix *M*

    It also provides recommendations and predictions using the reconstructed matrix *M'*

    :param filename: Path to a Zip file, containing an already computed SVD (U, Sigma, and V) for a matrix *M*
    :type filename: string
    """
    def __init__(self, filename=None):
        #Call parent constructor
        super(SVD, self).__init__()

        # self._U: Eigen vector. Relates the concepts of the input matrix to the principal axes
        # self._S (or \Sigma): Singular -or eigen- values. It represents the strength of each eigenvector.
        # self._V: Eigen vector. Relates features to the principal axes
        self._U, self._S, self._V = (None, None, None)
        # Mean centered Matrix: row and col shifts
        self._shifts = None
        # self._matrix_reconstructed: M' = U S V^t
        self._matrix_reconstructed = None

        # Similarity matrix: (U \Sigma)(U \Sigma)^T = U \Sigma^2 U^T
        # U \Sigma is concept_axes weighted by axis_weights.
        self._matrix_similarity = SimilarityMatrix()

        if filename:
            self.load_model(filename)

        # Row and Col ids. Only when importing from SVDLIBC
        self._file_row_ids = None
        self._file_col_ids = None

        #Update feature

    def __repr__(self):
        try:
            s = '\n'.join(('M\':' + str(self._reconstruct_matrix()), \
                'A row (U):' + str(self._reconstruct_matrix().right[1]), \
                'A col (V):' + str(self._reconstruct_matrix().left[1])))
        except TypeError:
            s = self._data.__repr__()
        return s

    def load_model(self, filename):
        """
        Loads SVD transformation (U, Sigma and V matrices) from a ZIP file

        :param filename: path to the SVD matrix transformation (a ZIP file)
        :type filename: string
        """
        try:
            zip = zipfile.ZipFile(filename, allowZip64=True)
        except:
            zip = zipfile.ZipFile(filename + '.zip', allowZip64=True)
        # Options file
        options = dict()
        for line in zip.open('README'):
            data = line.strip().split('\t')
            options[data[0]] = data[1]
        try:
            k = int(options['k'])
        except:
            k = 100 #TODO: nasty!!!

        # Load U, S, and V
        """
        #Python 2.6 only:
        #self._U = loads(zip.open('.U').read())
        #self._S = loads(zip.open('.S').read())
        #self._V = loads(zip.open('.V').read())
        """
        try:
            self._U = loads(zip.read('.U'))
        except:
            matrix = fromfile(zip.extract('.U', TMPDIR))
            vectors = []
            i = 0
            while i < len(matrix) / k:
                v = DenseVector(matrix[k*i:k*(i+1)])
                vectors.append(v)
                i += 1
            try:
                idx = [ int(idx.strip()) for idx in zip.read('.row_ids').split('\n') if idx]
            except:
                idx = [ idx.strip() for idx in zip.read('.row_ids').split('\n') if idx]
            #self._U = DenseMatrix(vectors) 
            self._U = DenseMatrix(vectors, OrderedSet(idx), None)
        try:
            self._V = loads(zip.read('.V'))
        except:
            matrix = fromfile(zip.extract('.V', TMPDIR))
            vectors = []
            i = 0
            while i < len(matrix) / k:
                v = DenseVector(matrix[k*i:k*(i+1)])
                vectors.append(v)
                i += 1
            try:
                idx = [ int(idx.strip()) for idx in zip.read('.col_ids').split('\n') if idx]
            except:
                idx = [ idx.strip() for idx in zip.read('.col_ids').split('\n') if idx]
            #self._V = DenseMatrix(vectors) 
            self._V = DenseMatrix(vectors, OrderedSet(idx), None)

        self._S = loads(zip.read('.S'))

        # Shifts for Mean Centerer Matrix
        self._shifts = None
        if '.shifts.row' in zip.namelist():
            self._shifts = [loads(zip.read('.shifts.row')), 
                            loads(zip.read('.shifts.col')),
                            loads(zip.read('.shifts.total'))
                           ]
        self._reconstruct_matrix(shifts=self._shifts, force=True)
        self._reconstruct_similarity(force=True)

    def save_model(self, filename, options={}):
        """
        Saves SVD transformation (U, Sigma and V matrices) to a ZIP file

        :param filename: path to save the SVD matrix transformation (U, Sigma and V matrices)
        :type filename: string
        :param options: a dict() containing the info about the SVD transformation. E.g. {'k': 100, 'min_values': 5, 'pre_normalize': None, 'mean_center': True, 'post_normalize': True}
        :type options: dict
        """
        if VERBOSE:
            sys.stdout.write('Saving svd model to %s\n' % filename)

        f_opt = open(filename + '.config', 'w')
        for option, value in options.items():
            f_opt.write('\t'.join((option, str(value))) + '\n')
        f_opt.close()
        # U, S, and V
        MAX_VECTORS = 2**21
        if len(self._U) < MAX_VECTORS:
            self._U.dump(filename + '.U')
        else:
            self._U.tofile(filename + '.U')
        if len(self._V) < MAX_VECTORS:
            self._V.dump(filename + '.V')
        else:
            self._V.tofile(filename + '.V')
        self._S.dump(filename + '.S')

        # Shifts for Mean Centered Matrix
        if self._shifts:
            #(row_shift, col_shift, total_shift)
            self._shifts[0].dump(filename + '.shifts.row')
            self._shifts[1].dump(filename + '.shifts.col')
            self._shifts[2].dump(filename + '.shifts.total')

        zip = filename
        if not filename.endswith('.zip') and not filename.endswith('.ZIP'):
            zip += '.zip'
        fp = zipfile.ZipFile(zip, 'w', allowZip64=True)

        # Store Options in the ZIP file
        fp.write(filename=filename + '.config', arcname='README')
        os.remove(filename + '.config')
        
        # Store matrices in the ZIP file
        for extension in ['.U', '.S', '.V']:
            fp.write(filename=filename + extension, arcname=extension)
            os.remove(filename + extension)

        # Store mean center shifts in the ZIP file
        if self._shifts:
            for extension in ['.shifts.row', '.shifts.col', '.shifts.total']:
                fp.write(filename=filename + extension, arcname=extension)
                os.remove(filename + extension)

        # Store row and col ids file, if importing from SVDLIBC
        if self._file_row_ids:
            fp.write(filename=self._file_row_ids, arcname='.row_ids')
        if self._file_col_ids:
            fp.write(filename=self._file_col_ids, arcname='.col_ids')


    def _reconstruct_similarity(self, post_normalize=True, force=True):
        if not self.get_matrix_similarity() or force:
            self._matrix_similarity = SimilarityMatrix()
            self._matrix_similarity.create(self._U, self._S, post_normalize=post_normalize)
        return self._matrix_similarity

    def _reconstruct_matrix(self, shifts=None, force=True):
        if not self._matrix_reconstructed or force:
            if shifts:
                self._matrix_reconstructed = divisi2.reconstruct(self._U, self._S, self._V, shifts=shifts)
            else:
                self._matrix_reconstructed = divisi2.reconstruct(self._U, self._S, self._V)
        return self._matrix_reconstructed

    def compute(self, k=100, min_values=None, pre_normalize=None, mean_center=False, post_normalize=True, savefile=None):
        """
        Computes SVD on matrix *M*, :math:`M = U \Sigma V^T`

        :param k: number of dimensions
        :type k: int
        :param min_values: min. number of non-zeros (or non-empty values) any row or col must have
        :type min_values: int
        :param pre_normalize: normalize input matrix. Possible values are tfidf, rows, cols, all.
        :type pre_normalize: string
        :param mean_center: centering the input matrix (aka mean substraction)
        :type mean_center: Boolean
        :param post_normalize: Normalize every row of :math:`U \Sigma` to be a unit vector. Thus, row similarity (using cosine distance) returns :math:`[-1.0 .. 1.0]`
        :type post_normalize: Boolean
        :param savefile: path to save the SVD factorization (U, Sigma and V matrices)
        :type savefile: string
        """
        super(SVD, self).compute(min_values) #creates matrix and does squish to not have empty values

        if VERBOSE:
            sys.stdout.write('Computing svd k=%s, min_values=%s, pre_normalize=%s, mean_center=%s, post_normalize=%s\n' 
                            % (k, min_values, pre_normalize, mean_center, post_normalize))
            if not min_values:
                sys.stdout.write('[WARNING] min_values is set to None, meaning that some funky recommendations might appear!\n')

        # Get SparseMatrix
        matrix = self._matrix.get()

        # Mean center?
        shifts, row_shift, col_shift, total_shift = (None, None, None, None)
        if mean_center:
            if VERBOSE:
                sys.stdout.write("[WARNING] mean_center is True. svd.similar(...) might return nan's. If so, then do svd.compute(..., mean_center=False)\n")
            matrix, row_shift, col_shift, total_shift = matrix.mean_center() 
            self._shifts = (row_shift, col_shift, total_shift)

        # Pre-normalize input matrix?
        if pre_normalize:
            """
            Divisi2 divides each entry by the geometric mean of its row norm and its column norm. 
            The rows and columns don't actually become unit vectors, but they all become closer to unit vectors.
            """
            if pre_normalize == 'tfidf':
                matrix = matrix.normalize_tfidf() #TODO By default, treats the matrix as terms-by-documents; 
                                                  # pass cols_are_terms=True if the matrix is instead documents-by-terms.
            elif pre_normalize == 'rows':
                matrix = matrix.normalize_rows()
            elif pre_normalize == 'cols':
                matrix = matrix.normalize_cols()
            elif pre_normalize == 'all':
                matrix = matrix.normalize_all()
            else:
                raise ValueError("Pre-normalize option (%s) is not correct.\n \
                                  Possible values are: 'tfidf', 'rows', 'cols' or 'all'" % pre_normalize)
        #Compute SVD(M, k)
        self._U, self._S, self._V = matrix.svd(k)
        # Sim. matrix = U \Sigma^2 U^T
        self._reconstruct_similarity(post_normalize=post_normalize, force=True)
        # M' = U S V^t
        self._reconstruct_matrix(shifts=self._shifts, force=True)

        if savefile:
            options = {'k': k, 'min_values': min_values, 'pre_normalize': pre_normalize, 'mean_center': mean_center, 'post_normalize': post_normalize}
            self.save_model(savefile, options)

    def _get_row_reconstructed(self, i, zeros=None):
        if zeros:
            return self._matrix_reconstructed.row_named(i)[zeros]
        return self._matrix_reconstructed.row_named(i)

    def _get_col_reconstructed(self, j, zeros=None):
        if zeros:
            return self._matrix_reconstructed.col_named(j)[zeros]
        return self._matrix_reconstructed.col_named(j)

    def predict(self, i, j, MIN_VALUE=None, MAX_VALUE=None):
        """
        Predicts the value of :math:`M_{i,j}`, using reconstructed matrix :math:`M^\prime = U \Sigma_k V^T`

        :param i: row in M, :math:`M_{i \cdot}`
        :type i: user or item id
        :param j: col in M, :math:`M_{\cdot j}`
        :type j: item or user id
        :param MIN_VALUE: min. value in M (e.g. in ratings[1..5] => 1)
        :type MIN_VALUE: float
        :param MAX_VALUE: max. value in M (e.g. in ratings[1..5] => 5)
        :type MAX_VALUE: float
        """
        if not self._matrix_reconstructed:
            self.compute() #will use default values!
        predicted_value = self._matrix_reconstructed.entry_named(i, j) #M' = U S V^t
        if MIN_VALUE:
            predicted_value = max(predicted_value, MIN_VALUE)
        if MAX_VALUE:
            predicted_value = min(predicted_value, MAX_VALUE)
        return float(predicted_value)

    def recommend(self, i, n=10, only_unknowns=False, is_row=True):
        """
        Recommends items to a user (or users to an item) using reconstructed matrix :math:`M^\prime = U \Sigma_k V^T`

        E.g. if *i* is a row and *only_unknowns* is True, it returns the higher values of :math:`M^\prime_{i,\cdot}` :math:`\\forall_j{M_{i,j}=\emptyset}`

        :param i: row or col in M
        :type i: user or item id
        :param n: number of recommendations to return
        :type n: int
        :param only_unknowns: only return unknown values in *M*? (e.g. items not rated by the user)
        :type only_unknowns: Boolean
        :param is_row: is param *i* a row (or a col)?
        :type is_row: Boolean
        """
        if not self._matrix_reconstructed:
            self.compute() #will use default values!
        item = None
        zeros = []
        if only_unknowns and not self._matrix.get():
            raise ValueError("Matrix is empty! If you loaded an SVD model you can't use only_unknowns=True, unless svd.create_matrix() is called")
        if is_row:
            if only_unknowns:
                zeros = self._matrix.get().row_named(i).zero_entries()
            item = self._get_row_reconstructed(i, zeros)
        else:
            if only_unknowns:
                zeros = self._matrix.get().col_named(i).zero_entries()
            item = self._get_col_reconstructed(i, zeros)
        return item.top_items(n)

    def load_updateDataTuple_foldin(self, filename, force=True, sep='\t', format={'value':0, 'row':1, 'col':2}, pickle=False,is_row=True,truncate=False):
        """
        Loads a dataset file that contains a SINGLE tuple (a dataset for a single user OR item , has to be either same row or same column depending on is_row aka tuple)

        See params definition in *datamodel.Data.load()*
        """
        # nDimension
        if force:
            self._updateData = Data()

        self._updateData.load(filename, force, sep, format, pickle)

        if VERBOSE:
            print "reading the new tuple"
        if(is_row):
            nDimensionLabels=self._V.all_labels()[0] #get labels from V matrix to complete the sparse matrix
            print type(nDimensionLabels)
            print type(nDimensionLabels[0])
            print len(nDimensionLabels)
            self._singleUpdateMatrix.create(self._updateData.get(), col_labels=nDimensionLabels, foldin=True,truncate=truncate)

        else:
            nDimensionLabels = self._U.all_labels() #get labels from U matrix to complete the sparse matrix
            print nDimensionLabels
            self._singleUpdateMatrix.create(self._updateData.get(), row_labels=nDimensionLabels, foldin=True,truncate=truncate)

        if not truncate:
            additionalElements=self._singleUpdateMatrix.get_additional_elements()
            #If it's trying to foldin a new user who has rated a new item which was not used before, then foldin the item first then foldin that user
            print "dimension",len(nDimensionLabels)
            print "additional elements:",additionalElements
            print "length",len(additionalElements)
            if len(additionalElements) !=0:
                for item in additionalElements:
                    if (is_row): #if I am folding in a row then , the additionals added that shouldn't be are the columns to be folded in to the rows
                        self._singleAdditionalFoldin.create([(0,nDimensionLabels[0],item)], row_labels=self._U.all_labels()[0])
                    else:
                        self._singleAdditionalFoldin.create([(0,item,nDimensionLabels[0])], col_labels=self._V.all_labels()[0])
                    self._update(update_matrix=self._singleAdditionalFoldin,is_row=not is_row)

        # #update the data matrix
        if VERBOSE:
            print "updating the sparse matrix"
        # print "matrix before update:",self._matrix.get().shape
        if self._matrix.get(): #if matrix not there due to load ignore it
            self._matrix.update(self._singleUpdateMatrix) # updating the data matrix for the zeroes , also for saving the data matrix if needed
        # print "matrix after update:",self._matrix.get().shape
        self._update(is_row=is_row)

    def _construct_batch_dictionary(self,data,is_row=True):
        '''
        
        :param data: Data()
        :param is_row: Boolean
        :return: constructs a dictionary with the row or col as the keys (depending on which is being added) with values as the tuples
        in self._batchDict
        '''
        # self._values = map(itemgetter(0), data)
        # self._rows = map(itemgetter(1), data)
        # self._cols = map(itemgetter(2), data)
        key_idx=1 #key index default is the row
        if not is_row:
            key_idx=2

        #collecting the significant col or row tuples at one place to fold them in at once

        for item in data: #data is a list of tuples so item is 1 tuple
            try:
                self._batchDict[item[key_idx]].append(item)
            except KeyError:
                self._batchDict[item[key_idx]] = []
                self._batchDict[item[key_idx]].append(item)

        #batch loaded , now need to fold them in one by one
        print "Batch loaded successfully"


    def load_updateDataBatch_foldin(self, filename, force=True, sep='\t', format={'value': 0, 'row': 1, 'col': 2},
                                 pickle=False, is_row=True,truncate=False):
            """
            Dont forget future work in presentation , remove old and insert new
            Loads a dataset file that contains Multiple tuples

            truncate:boolean-> sometimes new users rate new items not in the original SVD matrix so would you like new items to be truncated or folded in ? default is foldin
            is_row: boolean -> are you trying to foldin a row or a column ? yes->foldin row , no->foldin column
            See params definition in *datamodel.Data.load()*
            
            """
            # call update here until it finishes
            # nDimension
            if force:
                self._updateData = Data()

            self._updateData.load(filename, force, sep, format, pickle) #load array of tuples
            print "Reading the new batch"

            self._construct_batch_dictionary(self._updateData.get(),is_row)

            print "Folding in batch entries"
            nDimensionLabels=None
            if (is_row):
                nDimensionLabels = self._V.all_labels()[0]  # get labels from V matrix to complete the sparse matrix
                # print nDimensionLabels
            else:
                nDimensionLabels = self._U.all_labels()[0]  # get labels from U matrix to complete the sparse matrix
                # print nDimensionLabels
            length_of_dict=len(self._batchDict)
            i=0
            isbatch=True
            for key_idx in self._batchDict: #data in batchDict in form {key:[(tuple)]}
                print "user:",key_idx
                i += 1
                if (is_row):
                    self._singleUpdateMatrix.create(self._batchDict[key_idx], col_labels=nDimensionLabels,foldin=True,truncate=truncate)

                else:
                    self._singleUpdateMatrix.create(self._batchDict[key_idx], row_labels=nDimensionLabels,foldin=True,truncate=truncate)

                # if(i==length_of_dict):
                #     isbatch=False


                # If it's trying to foldin a new user who has rated a new item which was not used before, then foldin the item first then foldin that user
                if not truncate:
                    additionalElements = self._singleUpdateMatrix.get_additional_elements()
                    print "dimension", len(nDimensionLabels)
                    print "additional elements:", additionalElements
                    print "length", len(additionalElements)
                    if len(additionalElements) != 0:
                        for item in additionalElements:
                            if (is_row):  # if I am folding in a row then , the additionals added that shouldn't be are the columns to be folded in to the rows
                                self._singleAdditionalFoldin.create([(0, nDimensionLabels[0], item)],
                                                                    row_labels=self._U.all_labels()[0])
                            else:
                                self._singleAdditionalFoldin.create([(0, item, nDimensionLabels[0])],
                                                                    col_labels=self._V.all_labels()[0])
                            self._update(update_matrix=self._singleAdditionalFoldin, is_row=not is_row)


                # #update the data matrix
                print "updating the sparse matrix"
                # print "matrix before update:",self._matrix.get().shape
                if self._matrix.get(): #if matrix not there due to load ignore it
                    self._matrix.update(
                        self._singleUpdateMatrix,is_batch=isbatch)  # updating the data matrix for the zeroes , also for saving the data matrix if needed
                # print "matrix after update:",self._matrix.get().shape
                self._update(is_row=is_row,is_batch=isbatch) #Do foldin on the singleUpdateMatrix tuple

            self.update_sparse_matrix_data(is_batch=True)


    def update_sparse_matrix_data(self,squishFactor=10,is_batch=False):
        #update the data matrix
        # print "matrix before update:",self._matrix.get().shape
        if is_batch:
            if self._matrix.get():
                if VERBOSE:
                    print "updating sparse index"
                self._matrix.index_sparseMatrix()
            if VERBOSE:
                print "before updating, M=", self._matrix_reconstructed.shape
            # Sim. matrix = U \Sigma^2 U^T
            self._reconstruct_similarity(post_normalize=False, force=True)
            # M' = U S V^t
            self._reconstruct_matrix(shifts=self._shifts, force=True)
            if VERBOSE:
                print "done updating, M=", self._matrix_reconstructed.shape

        if self._matrix.get(): #if loaded model there is no matrix
            if VERBOSE:
                print "commiting the sparse data matrix by removing empty rows and columns divisi created"
            self._matrix.squish(squishFactor) # updating the data matrix for the zeroes ,#NOTE: Intensive so do at end
            # print "matrix after update:",self._matrix.get().shape


    def _update(self,update_matrix=None,is_row=True,is_batch=False): #update(tuple:denseVector tuple,isRow=True,,
      if VERBOSE:
          print "type of S",type(self._S)
          print "type of U",type(self._U)
          print "type of V",type(self._V)
          print "type of data",type(self._data)
          print "type of matrix",type(self._matrix)
          print "type of matrix reconstructed",type(self._matrix_reconstructed)
          print "type of matrix similarity",type(self._matrix_similarity)

          print "dimensions of S",self._S.shape
          print "dimensions of U",self._U.shape
          print "dimensions of V",self._V.shape

      invS=np.zeros((self._S.shape[0], self._S.shape[0]))
      for i in range(self._S.shape[0]):
          # invS[i, i] = self._S[i]  # creating diagonal matrix
          invS[i, i] = self._S[i]**-1  # creating diagonal matrix and inverting using special property of diagonal matrix
      # invS=inv(invS) inverting with numpy

      #if new is row -> V*S^-1
      if is_row:
        prodM=self._V.dot(invS)
        if VERBOSE:
            print "dimension of VxS^-1=", prodM.shape
      else:       #if new is col -> U*S^-1
        prodM = self._U.dot(invS)
        if VERBOSE:
            print "dimension of UxS^-1=", prodM.shape

      if update_matrix:
          updateTupleMatrix=update_matrix.get()
      else:
          updateTupleMatrix = self._singleUpdateMatrix.get()

      if not is_row:
          updateTupleMatrix=updateTupleMatrix.transpose() #transpose
      if VERBOSE:
          print "dimensions of user",updateTupleMatrix.shape
      res=updateTupleMatrix.dot(prodM)
      if VERBOSE:
          print "type of res=", type(res)
          print "dimension of resultant is", res.shape

      if is_row:
      #use new value can now be concatinated with U
        if VERBOSE:
            print "U before adding", self._U.shape
        self._U=self._U.concatenate(res)
        if VERBOSE:
            print "U after adding", self._U.shape

      else:
        if VERBOSE:
            print "V before adding", self._V.shape
        self._V = self._V.concatenate(res)
        if VERBOSE:
            print "V after adding", self._V.shape

     #TODO: contemplating removing this segment and just reconstruct in the updating spare matrix function
      if not is_batch: #will reconstruct all at end with batch using another function
        if VERBOSE:
            print "before updating, M=",self._matrix_reconstructed.shape
        # Sim. matrix = U \Sigma^2 U^T
        self._reconstruct_similarity(post_normalize=False, force=True)
        # M' = U S V^t
        self._reconstruct_matrix(shifts=self._shifts, force=True)
        if VERBOSE:
            print "done updating, M=",self._matrix_reconstructed.shape




      # myFile=open("prodMVSq.dat",'w')
      # myFile.truncate()
      #
      # for i in range(20):
      #   myFile.write(str(res[0, i])+" ")
      #
      #   myFile.write("\n")

      # # invS = inv(diag_S)
      # # print "dimensions of S^-1", invS.shape
      #
      #
      # print "writing s to file"
      # myFile=open("invS.dat",'w')
      # myFile.truncate()
      # # for item in self.invS.tolist():
      # #     myFile.write(str(item))
      # #     myFile.write("\n")
      # myFile.write("dimensions= "+str(invS.shape))
      # myFile.write("\n")
      # for i in range(invS.shape[0]):
      #   myFile.write(str(invS[i,i]))
      #   myFile.write("\n")

    def printMovies(self):
        myFile=open("movieIDs.dat",'w')
        myFile.truncate()

        movies=self._matrix_reconstructed.get_col_labels()
        for movie in movies :
          myFile.write(str(movie)+",")

    def centroid(self, ids, is_row=True):
        points = []
        for id in ids:
            if is_row:
                point = self._U.row_named(id)
            else:
                point = self._V.row_named(id)
            points.append(point)
        M = divisi2.SparseMatrix(points)
        return M.col_op(sum)/len(points) #TODO Numpy.sum?

    def kmeans(self, ids, k=5, components=3, are_rows=True):
        """
        K-means clustering. It uses k-means++ (http://en.wikipedia.org/wiki/K-means%2B%2B) to choose the initial centroids of the clusters

        Clusterizes a list of IDs (either row or cols)

        :param ids: list of row (or col) ids to cluster
        :param k: number of clusters
        :param components: how many eigen values use (from SVD)
        :param are_rows: is param *ids* a list of rows (or cols)?
        :type are_rows: Boolean
        """
        if not isinstance(ids, list):
            # Cluster the whole row(or col) values. It's slow!
            return super(SVD, self).kmeans(ids, k=k, is_row=are_rows)
        if VERBOSE:
            sys.stdout.write('Computing k-means, k=%s for ids %s\n' % (k, ids))
        MAX_POINTS = 150
        points = []
        for id in ids:
            if are_rows:
                points.append(self._U.row_named(id)[:components])
            else:
                points.append(self._V.row_named(id)[:components])
        M = array(points)
        # Only apply Matrix initialization if num. points is not that big!
        if len(points) <= MAX_POINTS:
            centers = self._kinit(array(points), k)
            centroids, labels = kmeans2(M, centers, minit='matrix')
        else:
            centroids, labels = kmeans2(M, k, minit='random')
        i = 0
        clusters = dict()
        for cluster in labels:
            if not clusters.has_key(cluster): 
                clusters[cluster] = dict()
                clusters[cluster]['centroid'] = centroids[cluster]
                clusters[cluster]['points'] = []
            point = self._U.row_named(ids[i])[:components]
            centroid = clusters[cluster]['centroid']
            to_centroid = self._cosine(centroid, point)
            clusters[cluster]['points'].append((ids[i], to_centroid))
            clusters[cluster]['points'].sort(key=itemgetter(1), reverse=True)
            i += 1
        return clusters

    '''
    def kmeans(self, id, k=5, is_row=True):
        """
        K-means clustering. It uses k-means++ (http://en.wikipedia.org/wiki/K-means%2B%2B) for choosing the initial centroids of the clusters

        Clusterizes the (cols) values of a given row, or viceversa

        :param id: row (or col) id to cluster its values
        :param k: number of clusters
        :param is_row: is param *id* a row (or a col)?
        :type is_row: Boolean
        """
        if VERBOSE:
            sys.stdout.write('Computing k-means (from SVD) for %s, with k=%s\n' % (id, k))
        point = None
        if is_row:
            point = self.get_matrix().get_row(id)
        else:
            point = self.get_matrix().get_col(id)
        points = []
        for i in point.nonzero_entries():
            label = point.label(i)
            points.append(label)
        return self._kmeans(points, k, not is_row)
    '''

# SVDNeighbourhood
class SVDNeighbourhood(SVD):
    """
    Classic Neighbourhood plus Singular Value Decomposition. Inherits from SVD class

    Predicts the value of :math:`M_{i,j}`, using simple avg. (weighted) of
    all the ratings by the most similar users (or items). This similarity, *sim(i,j)* is derived from the SVD

    :param filename: Path to a Zip file, containing an already computed SVD (U, Sigma, and V) for a matrix *M*
    :type filename: string
    :param Sk: number of similar elements (items or users) to be used in *predict(i,j)*
    :type Sk: int
    """
    def __init__(self, filename=None, Sk=10):
        # Call parent constructor
        super(SVDNeighbourhood, self).__init__(filename)

        # Number of similar elements
        self._Sk = Sk #Length of Sk(i;u)

    def similar_neighbours(self, i, j, Sk=10):
        similars = self.similar(i, Sk*10) #Get 10 times Sk
        # Get only those items that user j has already rated
        current = 0
        _Sk = Sk
        for similar, weight in similars[1:]:
            if self.get_matrix().value(similars[current][0], j) == 0.0:
                similars.pop(current)
                current -= 1
                _Sk += 1
            current += 1
            _Sk -= 1
            if _Sk == 0: 
                break # We have enough elements to use
        return similars[:Sk]

    def predict(self, i, j, Sk=10, weighted=True, MIN_VALUE=None, MAX_VALUE=None):
        """
        Predicts the value of :math:`M_{i,j}`, using simple avg. (weighted) of
        all the ratings by the most similar users (or items)

        if *weighted*:
            :math:`\hat{r}_{ui} = \\frac{\sum_{j \in S^{k}(i;u)} sim(i, j) r_{uj}}{\sum_{j \in S^{k}(i;u)} sim(i, j)}`

        else:
            :math:`\hat{r}_{ui} = mean(\sum_{j \in S^{k}(i;u)} r_{uj})`

        :param i: row in M, :math:`M_{i \cdot}`
        :type i: user or item id
        :param j: col in M, :math:`M_{\cdot j}`
        :type j: item or user id
        :param Sk: number of k elements to be used in :math:`S^k(i; u)`
        :type Sk: int
        :param weighted: compute avg. weighted of all the ratings?
        :type weighted: Boolean
        :param MIN_VALUE: min. value in M (e.g. in ratings[1..5] => 1)
        :type MIN_VALUE: float
        :param MAX_VALUE: max. value in M (e.g. in ratings[1..5] => 5)
        :type MAX_VALUE: float
        """
        if not Sk:
            Sk = self._Sk
        similars = self.similar_neighbours(i, j, Sk)
        #Now, similars == S^k(i; u)

        sim_ratings = []
        sum_similarity = 0.0
        for similar, weight in similars:
            sim_rating = self.get_matrix().value(similar, j)
            if sim_rating is None: #== 0.0:
                continue
            sum_similarity += weight
            if weighted:
                sim_ratings.append(weight * sim_rating)
            else:
                sim_ratings.append(sim_rating)

        if not sum_similarity or not sim_ratings:
            return nan

        if weighted:
            predicted_value = sum(sim_ratings)/sum_similarity
        else:
            predicted_value = mean(sim_ratings)
        if MIN_VALUE:
            predicted_value = max(predicted_value, MIN_VALUE)
        if MAX_VALUE:
            predicted_value = min(predicted_value, MAX_VALUE)
        return float(predicted_value)


# SVDNeighbourhoodKoren
class __SVDNeighbourhoodKoren(SVDNeighbourhood):
    """
    Inherits from SVDNeighbourhood class. 

    Neighbourhood model, using Singular Value Decomposition.
    Based on 'Factorization Meets the Neighborhood: a Multifaceted
    Collaborative Filtering Model' (Yehuda Koren)
    http://public.research.att.com/~volinsky/netflix/kdd08koren.pdf

    :param filename: Path to a Zip file, containing an already computed SVD (U, Sigma, and V) for a matrix *M*
    :type filename: string
    :param Sk: number of similar elements (items or users) to be used in *predict(i,j)*
    :type Sk: int
    """
    def __init__(self, filename=None, Sk=10):
        # Call parent constructor
        super(SVDNeighbourhoodKoren, self).__init__(filename, Sk)

        # µ denotes the overall average rating
        self._Mu = None
        # Mean of all rows
        self._mean_rows = None
        # Mean of all cols
        self._mean_cols = None
        # Mean of each row / col
        self._mean_row = dict()
        self._mean_col = dict()

    def set_mu(self, mu):
        """
        Sets the :math:`\mu`. The overall average rating

        :param mu: overall average rating
        :type mu: float
        """
        self._Mu = mu

    def _set_mean_all(self, avg=None, is_row=True):
        m = self._mean_row.values()
        if not is_row:
            m = self._mean_col.values()
        return mean(m)

    def set_mean_rows(self, avg=None):
        """
        Sets the average value of all rows

        :param avg: the average value (if None, it computes *average(i)*)
        :type avg: float
        """
        self._mean_rows = self._set_mean_all(avg, is_row=True)

    def set_mean_cols(self, avg=None):
        """
        Sets the average value of all cols

        :param avg: the average value (if None, it computes *average(i)*)
        :type avg: float
        """
        self._mean_cols = self._set_mean_all(avg, is_row=False)

    def set_mean(self, i, avg=None, is_row=True):
        """
        Sets the average value of a row (or column).

        :param i: a row (or column)
        :type i: user or item id
        :param avg: the average value (if None, it computes *average(i)*)
        :type avg: float
        :param is_row: is param *i* a row (or a col)?
        :type is_row: Boolean
        """
        d = self._mean_row
        if not is_row:
            d = self._mean_col
        if avg is None: #Compute average
            m = self._matrix.get().row_named
            if not is_row:
                m = self._matrix.get().col_named
            avg = mean(m(i))
        d[i] = avg

    def predict(self, i, j, Sk=None, MIN_VALUE=None, MAX_VALUE=None):
        """
        Predicts the value of *M(i,j)*

        It is based on 'Factorization Meets the Neighborhood: a Multifaceted
        Collaborative Filtering Model' (Yehuda Koren). 
        Equation 3 (section 2.2):

        :math:`\hat{r}_{ui} = b_{ui} + \\frac{\sum_{j \in S^k(i;u)} s_{ij} (r_{uj} - b_{uj})}{\sum_{j \in S^k(i;u)} s_{ij}}`, where
        :math:`b_{ui} = \mu + b_u + b_i`

        http://public.research.att.com/~volinsky/netflix/kdd08koren.pdf

        :param i: row in M, M(i)
        :type i: user or item id
        :param j: col in M, M(j)
        :type j: user or item id
        :param Sk: number of k elements to be used in :math:`S^k(i; u)`
        :type Sk: int
        :param MIN_VALUE: min. value in M (e.g. in ratings[1..5] => 1)
        :type MIN_VALUE: float
        :param MAX_VALUE: max. value in M (e.g. in ratings[1..5] => 5)
        :type MAX_VALUE: float

        """
        # http://public.research.att.com/~volinsky/netflix/kdd08koren.pdf
        # bui = µ + bu + bi
        #   The parameters bu and bi indicate the observed deviations of user
        #   u and item i, respectively, from the average
        # 
        # S^k(i; u): 
        #   Using the similarity measure, we identify the k items rated
        #   by u, which are most similar to i.
        #
        # sij: similarity between i and j
        #
        # r^ui = bui + Sumj∈S^k(i;u) sij (ruj − buj) / Sumj∈S^k(i;u) sij
        if not Sk:
            Sk = self._Sk

        similars = self.similar_neighbours(i, j, Sk)
        #Now, similars == S^k(i; u)

        #bu = self._mean_col.get(j, self.set_mean(j, is_row=False)) - self._mean_cols
        #bi = self._mean_row.get(i, self.set_mean(i, is_row=True)) - self._mean_rows
        bu = self._mean_col[j] - self._mean_cols
        bi = self._mean_row[i] - self._mean_rows
        bui = bu + bi
        #if self._Mu: #TODO uncomment?
        #   bui += self._Mu
 
        sim_ratings = []
        sum_similarity = 0.0
        for similar, sij in similars[1:]:
            sim_rating = self.get_matrix().value(similar, j)
            if sim_rating is None:
                continue
            ruj = sim_rating
            sum_similarity += sij
            bj = self._mean_row[similar]- self._mean_rows
            buj = bu + bj
            sim_ratings.append(sij * (ruj - buj))

        if not sum_similarity or not sim_ratings:
            return nan

        Sumj_Sk = sum(sim_ratings)/sum_similarity
        rui = bui + Sumj_Sk
        predicted_value = rui
        
        if MIN_VALUE:
            predicted_value = max(predicted_value, MIN_VALUE)
        if MAX_VALUE:
            predicted_value = min(predicted_value, MAX_VALUE)
        return float(predicted_value)

