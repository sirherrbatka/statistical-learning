(cl:in-package #:statistical-learning.gradient-descent-refine)


(defgeneric refine-trees (tree-parameters algorithm
                          ensemble train-data target-data))
