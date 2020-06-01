(cl:in-package #:cl-grf.performance)


(defgeneric performance-metric (parameters target predictions))
(defgeneric average-performance-metric (parameters metrics))
(defgeneric errors (parameters target predictions))
