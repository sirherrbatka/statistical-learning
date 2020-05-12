(cl:in-package #:cl-grf.model-protocol)


(defclass fundamental-model ()
  ())


(defclass fundamental-model-parameters ()
  ())


(defclass fundamental-performance-metric ()
  ())


(defclass confusion-matrix (fundamental-performance-metric)
  ((%positive :initarg :total-positive
              :reader positive)
   (%true-positive :initarg :true-positive)
   (%negative :initarg :total-negative
              :reader negative)
   (%true-negative :initarg :true-negative
                   :reader true-negative)))
