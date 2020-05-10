(cl:in-package #:cl-grf.algorithms)


(defgeneric minimal-difference (training-parameters))
(defgeneric score (object))
(defgeneric (setf score) (new-value object))
(defgeneric calculate-score (training-parameters split-array target-data))
(defgeneric predictions (leaf))
(defgeneric support (leaf))
(defgeneric (setf support) (new-value leaf))
(defgeneric (setf predictions) (new-value leaf))
