(cl:in-package #:statistical-learning.omp)


(defclass orthogonal-matching-pursuit (sl.ensemble:fundamental-pruning-algorithm)
  ((%number-of-trees-selected :initarg :number-of-trees-selected
                              :reader number-of-trees-selected)
   (%threshold :initarg :threshold
               :reader threshold))
  (:default-initargs
   :number-of-trees-selected nil
   :threshold nil))