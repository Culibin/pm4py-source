class ProcessTree(object):

    def __init__(self, operator=None, parent=None, children=None, label=None, index_c=None, place=None):
        """
        Constructor

        Parameters
        ------------
        operator
            Operator (of the current node) of the process tree
        parent
            Parent node (of the current node)
        children
            List of children of the current node
        label
            Label (of the current node)
        """
        self._operator = operator
        self._parent = parent
        self._children = list() if children is None else children
        self._label = label
        self._index_c = index_c
        self._place = place

    def _set__operator(self, operator):
        self._operator = operator

    def _set_parent(self, parent):
        self._parent = parent

    def _set_label(self, label):
        self._label = label

    def _set_index_c(self, index_c):
        self._index_c = index_c

    def _get_index_c(self):
        return self._index_c

    def _get_children(self):
        return self._children

    def _get_parent(self):
        return self._parent

    def _get_operator(self):
        return self._operator

    def _get_label(self):
        return self._label

    def __repr__(self):
        """
        Returns a string representation of the process tree

        Returns
        ------------
        stri
            String representation of the process tree
        """
        if self.operator is not None:
            rep = str(self._operator) + '( '
            for i in range(0, len(self._children)):
                child = self._children[i]
                rep += str(child) + ', ' if i < len(self._children) - 1 else str(child)
            return rep + ' )'
        elif self.label is not None:
            return self.label
        else:
            return u'\u03c4'

    def __str__(self):
        """
        Returns a string representation of the process tree

        Returns
        ------------
        stri
            String representation of the process tree
        """
        return self.__repr__()

    parent = property(_get_parent, _set_parent)
    children = property(_get_children)
    operator = property(_get_operator)
    label = property(_get_label, _set_label)
    index_c = property(_get_index_c)

    def count_nodes(self):

        count = 1
        for i in range(0, len(self._children)):
            child = self._children[i]
            count += child.count_nodes()
        return count

    def index_nodes(self, index_c):
        self._set_index_c(index_c)
        index_c += 1
        for i in range(0, len(self._children)):
            child = self._children[i]
            index_c = child.index_nodes(index_c)

        return index_c
