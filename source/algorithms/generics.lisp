(cl:in-package #:cl-grf.algorithms)


(defgeneric minimal-difference (training-parameters))
(defgeneric (setf minimal-difference) (new-value training-parameters))
(defgeneric score (object))
(defgeneric (setf score) (new-value object))
(defgeneric calculate-score (training-parameters split-array target-data))
(defgeneric predictions (leaf))
(defgeneric (setf predictions) (new-value leaf))
