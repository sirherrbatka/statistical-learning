(cl:in-package #:statistical-learning.omp)


(defclass orthogonal-matching-pursuit (sl.ensemble:fundamental-pruning-algorithm)
  ((%number-of-trees-selected :initarg :number-of-trees-selected
                              :reader number-of-trees-selected)))
