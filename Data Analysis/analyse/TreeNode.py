class treeNode():
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode  # needs to be updated
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(('  ' * ind, self.name, ' ', self.count))
        for child in self.children.values():
            child.disp(ind + 1)
