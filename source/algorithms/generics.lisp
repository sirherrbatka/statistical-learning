(cl:in-package #:cl-grf.algorithms)


(defgeneric shrinkage (gradient-boost))
(defgeneric minimal-difference (training-parameters))
(defgeneric score (object))
(defgeneric (setf score) (new-value object))
(defgeneric predictions (leaf))
(defgeneric (setf predictions) (new-value leaf))
(defgeneric support (leaf))
(defgeneric (setf support) (new-value node))
(defgeneric calculate-expected-value (parameters data))
(defgeneric calculate-score (parameters split-array target-data))
(defgeneric gradient-boost-response* (parameters expected gathered-predictions))
