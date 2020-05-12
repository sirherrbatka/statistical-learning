(cl:in-package #:cl-grf.model-protocol)


(defclass fundamental-model ()
  ())


(defclass fundamental-model-parameters ()
  ())


(defclass fundamental-performance-metric ()
  ())


(defclass confusion-matrix (fundamental-performance-metric)
  ((%total-positive :initarg :total-positive
                    :reader total-positive)
   (%true-positive :initarg :true-positive)
   (%total-negative :initarg :total-negative
                    :reader total-negative)
   (%true-negative :initarg :true-negative
                   :reader true-negative)))
