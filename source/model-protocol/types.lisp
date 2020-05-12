(cl:in-package #:cl-grf.model-protocol)


(defclass fundamental-model ()
  ())


(defclass fundamental-model-parameters ()
  ())


(defclass fundamental-performance-metric ()
  ())


(defclass confusion-matrix (fundamental-performance-metric)
  ((%positive :initarg :positive
              :reader positive)
   (%true-positive :initarg :positive
                   :reader true-positive)
   (%negative :initarg :negative
              :reader negative)
   (%true-negative :initarg :negative
                   :reader true-negative)))
