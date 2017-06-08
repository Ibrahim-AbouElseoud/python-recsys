try:
    from divisi2.sparse import SparseMatrix as divisiSparseMatrix
    from divisi2 import reconstruct_similarity
except:
    from csc.divisi2.sparse import SparseMatrix as divisiSparseMatrix
    from csc.divisi2 import reconstruct_similarity

from operator import itemgetter

class Matrix(object):
    def __init__(self):
        self._matrix = None

    def __repr__(self):
        return str(self._matrix)

    def create(self, data):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def density(self, percent=True):
        if not self._matrix or not self._matrix.entries():
            return None
        density = self._matrix.density()
        if percent:
            density *= 100
        return round(density, 4)

    def empty(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def get(self):
        return self._matrix

    def set(self, matrix):
        self._matrix = matrix

    def get_row(self, i):
        if self.empty() or not self._matrix.col_labels:
            raise ValueError('Matrix is empty (or has no columns!)')
        return self._matrix.row_named(i)

    def get_col(self, j):
        if self.empty() or not self._matrix.row_labels:
            raise ValueError('Matrix is empty (or has no rows!)')
        return self._matrix.col_named(j)

    def value(self, i, j):
        if self.empty():
            raise ValueError('Matrix is empty!')
        return self._matrix.entry_named(i, j)

    def get_value(self, i, j):
        if self.empty():
            raise ValueError('Matrix is empty!')
        return self.value(i, j)

    def set_value(self, i, j, value):
        if self.empty():
            raise ValueError('Matrix is empty!')
        self._matrix.set_entry_named(i, j, value)

    def get_row_len(self):
        if self.empty() or not self._matrix.col_labels:
            raise ValueError('Matrix is empty (or has no columns!)')
        return len(self._matrix.col_labels)

    def get_col_len(self):
        if self.empty() or not self._matrix.row_labels:
            raise ValueError('Matrix is empty (or has no rows!)')
        return len(self._matrix.row_labels)


class SparseMatrix(Matrix):
    def __init__(self):
        super(SparseMatrix, self).__init__()
        self._values=None
        self._rows=None
        self._cols=None
        self._additional_elements=[] #if no additional then len will equal 0


    def get_rows(self): #can use to get rated items and remove from recommendation
        return self._rows

    def get_cols(self): #can use to get rated items and remove from recommendation
        return self._cols

    def get_additional_elements(self):  # can use to get additional items to either fold or truncate
        return self._additional_elements

#row_labels specifies the row labels the complete matrix should have incase the inputted file doesn't include all indicies and it was saved in previous matrix (for update)
#same explination for col_labels but for columns
#matrix should have, in case it is larger than the largest index.
    def create(self, data,row_labels=None, col_labels=None, foldin=False,truncate=False):
         #is_row is what I'm originally folding in
        self._values = map(itemgetter(0), data)
        self._rows = map(itemgetter(1), data)
        self._cols = map(itemgetter(2), data)

        if foldin: #new to make sure not folding in user and item at same time
            #idea: create matrix normally but keep track of the columns (items) or rows to be folded in before doing update
            if col_labels: #if col_labels defined then I'm folding in a row
                self._additional_elements = [x for x in self._cols if x not in col_labels]
            else: #else I am folding in a column
                self._additional_elements = [x for x in self._rows if x not in row_labels]
            if truncate:
                for item in self._additional_elements:
                    if col_labels:
                        index_remove = self._cols.index(item)
                    else:
                        index_remove = self._rows.index(item)
                    del self._values[index_remove]
                    del self._rows[index_remove]
                    del self._cols[index_remove]


        self._matrix = divisiSparseMatrix.from_named_lists(self._values, self._rows, self._cols,row_labels, col_labels)




    def update(self, matrix,is_batch=False): #isbatch is for creating the final sparse matrix ,since you will want to collect all then construct final matrix at end
#To update the stored data matrix with the new values and create a new divisi spare matrix with it to retain the zeroes
        self._values.extend(matrix._values)
        self._rows.extend(matrix._rows)
        self._cols.extend(matrix._cols)

        if not is_batch:
            self._matrix = divisiSparseMatrix.from_named_lists(self._values, self._rows, self._cols)

    def squish(self,squishFactor): #remove additional empty fields created by divisiSparseMatrix
        self._matrix=self._matrix.squish(squishFactor)

    def index_sparseMatrix(self): #create the divisi2 sparse matrix from already existing values
        self._matrix = divisiSparseMatrix.from_named_lists(self._values, self._rows, self._cols)

    def empty(self):
        return not self._matrix or not self._matrix.values()

class SimilarityMatrix(Matrix):
    def __init__(self):
        super(SimilarityMatrix, self).__init__()

    def create(self, U, S, post_normalize=False):
        self._matrix = reconstruct_similarity(U, S, post_normalize=post_normalize)

    def empty(self):
        nrows, ncols = (0, 0)
        if self._matrix:
            nrows, ncols = self._matrix.shape
        return not self._matrix or not (nrows and ncols)
