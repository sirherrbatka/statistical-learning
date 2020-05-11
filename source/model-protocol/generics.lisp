(cl:in-package #:cl-grf.model-protocol)


(defgeneric make-model (parameters train-data target-data))
(defgeneric predict (model data))
(defgeneric performance-metric (parameters target predictions))
(defgeneric average-performance-metric (parameters metrics))
