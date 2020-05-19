(cl:in-package #:cl-grf.model-protocol)


(defgeneric make-model (parameters train-data target-data &optional weights))
(defgeneric predict (model data &optional parallel))
