(cl:in-package #:statistical-learning.ensemble)


(defgeneric trees (random-forest))
(defgeneric trees-count (random-forest-parameters))
(defgeneric tree-attributes-count (random-forest-parameters))
(defgeneric parallel (parameters))
(defgeneric tree-parameters (parameters))
(defgeneric tree-sample-rate (parameters))
(defgeneric tree-batch-size (parameters))
(defgeneric update-weights (calculator tree-parameters
                            prev-trees samples))
