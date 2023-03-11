(cl:in-package #:statistical-learning.struct-forest)


(defgeneric struct-target-data (state))
(defgeneric training-implementation (parameters &rest initargs))
(defgeneric prediction-implementation (parameters &rest initargs))
(defgeneric relable (parameters state))
