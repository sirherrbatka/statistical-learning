(cl:in-package #:cl-grf.forest)


(defgeneric forest-class (random-forest-parameters))
(defgeneric trees (random-forest))
(defgeneric attributes (random-forest))
(defgeneric leafs-for (random-forest data &optional parallel))
(defgeneric predictions-from-leafs (random-forest leafs &optional parallel))
(defgeneric trees-count (random-forest-parameters))
(defgeneric tree-attributes-count (random-forest-parameters))
(defgeneric parallel (parameters))
(defgeneric tree-parameters (parameters))
(defgeneric tree-sample-rate (parameters))
(defgeneric tree-batch-size (parameters))
(defgeneric attributes-importance (forest training-data target-data))
(defgeneric weights-calculator (training-parameters parallel weights
                                train-data target-data))
