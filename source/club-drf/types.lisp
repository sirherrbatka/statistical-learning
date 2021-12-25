(cl:in-package #:statistical-learning.club-drf)


(defclass club-drf (sl.ensemble:fundamental-pruning-algorithm)
  ((%parallel :initarg :parallel
              :initform nil
              :reader parallel)
   (%number-of-trees-selected :initarg :number-of-trees-selected
                              :reader number-of-trees-selected)
   (%max-neighbor :initarg :max-neighbor
                  :reader max-neighbor)))
