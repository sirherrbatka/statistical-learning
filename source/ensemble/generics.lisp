(cl:in-package #:statistical-learning.ensemble)


(defgeneric trees (ensemble))
(defgeneric trees-count (ensemble-parameters))
(defgeneric tree-attributes-count (ensemble-parameters))
(defgeneric parallel (parameters))
(defgeneric tree-parameters (parameters))
(defgeneric tree-sample-rate (parameters))
(defgeneric tree-batch-size (parameters))
(defgeneric update-weights (calculator tree-parameters
                            prev-trees samples))
(defgeneric leafs (ensemble data &optional parallel))
