(cl:in-package #:statistical-learning.performance)


(defgeneric performance-metric* (parameters type target predictions weights))
(defgeneric default-performance-metric (parameters))
(defgeneric average-performance-metric (parameters metrics))
(defgeneric errors (parameters target predictions))
