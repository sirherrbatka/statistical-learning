(cl:in-package #:statistical-learning.ensemble)


(defgeneric trees (ensemble))
(defgeneric trees-count (ensemble-parameters))
(defgeneric tree-attributes-count (ensemble-parameters))
(defgeneric parallel (parameters))
(defgeneric tree-parameters (parameters))
(defgeneric tree-sample-rate (parameters))
(defgeneric tree-batch-size (parameters))
(defgeneric update-weights (calculator tree-parameters
                            ensemble-state))
(defgeneric leafs (ensemble data &optional parallel))
(defgeneric assign-leafs (state))
(sl.common:defgeneric/proxy make-tree-training-state
    ((ensemble-parameters)
     (tree-parameters)
     ensemble-state
     attributes
     sample
     initargs))
(sl.common:defgeneric/proxy after-tree-fitting
    ((ensemble-parameters)
     (tree-parametres)
     ensemble-state))
