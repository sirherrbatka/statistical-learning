(cl:in-package #:statistical-learning.model-protocol)


(defgeneric make-model (parameters train-data target-data &key weights &allow-other-keys))
(defgeneric predict (model data &optional parallel))
(defgeneric parameters (model))
