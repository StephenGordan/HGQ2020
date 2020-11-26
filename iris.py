

## 鸢尾花，需要在jupyter notebook 中使用
## 在cmd 中输入 jupyter notebook 打开网页 new -> python3


import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.datasets import load_iris
import pydotplus
from IPython.display import Image, display

iris = load_iris()

clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf,
                                out_file = None,
                                feature_names = iris.feature_names,
                                class_names = iris.target_names,
                                filled = True,
                                rounded = True
                                )

graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))

