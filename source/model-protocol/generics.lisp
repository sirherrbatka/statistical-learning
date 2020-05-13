(cl:in-package #:cl-grf.model-protocol)


(defgeneric make-model (parameters train-data target-data))
(defgeneric predict (model data))
