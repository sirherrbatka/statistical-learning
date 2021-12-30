(cl:in-package #:statistical-learning.omp)


(defclass parameters (sl.ensemble:fundamental-pruning-algorithm)
  ((%number-of-trees-selected :initarg :number-of-trees-selected
                              :reader number-of-trees-selected)
   (%sample-size :initarg :sample-size
                 :initform nil
                 :reader sample-size)
   (%threshold :initarg :threshold
               :reader threshold))
  (:default-initargs
   :number-of-trees-selected nil
   :threshold nil))
