(cl:in-package #:cl-grf.performance)


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
