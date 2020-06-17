(cl:in-package #:statistical-learning.gradient-boost-tree)


(defgeneric implementation (parameters &rest initargs))
(defgeneric target (parameters target-data expected-value))
(defgeneric contributed-predictions (parameters model data-points-count))
(defgeneric calculate-expected-value (parameters target-data))
(defgeneric calculate-response (parameters predictions expected))
