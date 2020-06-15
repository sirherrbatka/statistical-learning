(cl:in-package #:statistical-learning.performance)


(defgeneric performance-metric (parameters target predictions &key weights))
(defgeneric average-performance-metric (parameters metrics))
(defgeneric errors (parameters target predictions))
