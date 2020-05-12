(cl:in-package #:cl-grf.model-protocol)


(defgeneric make-model (parameters train-data target-data))
(defgeneric predict (model data))
(defgeneric performance-metric (parameters target predictions))
(defgeneric average-performance-metric (parameters metrics))
(defgeneric positive (confusion-matrix))
(defgeneric true-positive (confusion-matrix))
(defgeneric negative (confusion-matrix))
(defgeneric true-negative (confusion-matrix))
