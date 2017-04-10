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

#`nrows` and `ncols` specify the shape the resulting
#matrix should have, in case it is larger than the largest index.
    def create(self, data,row_labels=None, col_labels=None):
        self._values = map(itemgetter(0), data)
        self._rows = map(itemgetter(1), data)
        self._cols = map(itemgetter(2), data)
        self._matrix = divisiSparseMatrix.from_named_lists(self._values, self._rows, self._cols,row_labels, col_labels)

    def update(self, matrix):

        self._values.extend(matrix._values)
        self._rows.extend(matrix._rows)
        self._cols.extend(matrix._cols)

        self._matrix = divisiSparseMatrix.from_named_lists(self._values, self._rows, self._cols)

    def squish(self,squishFactor):
        self._matrix=self._matrix.squish(squishFactor)



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

