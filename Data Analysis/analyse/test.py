import pandas as pd
ntest = ['a','b']
ltest = [[1,2], [4,5,6]]

data = [(k, v) for k, l in zip(ntest, ltest) for v in l]

hh =pd.DataFrame(data)
x=1