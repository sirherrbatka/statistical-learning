(cl:in-package #:statistical-learning.club-drf)


(defclass club-drf (sl.ensemble:fundamental-pruning-algorithm)
  ((%parallel :initarg :parallel
              :initform nil
              :reader parallel)
   (%number-of-trees-selected :initarg :number-of-trees-selected
                              :reader number-of-trees-selected)
   (%max-neighbor :initarg :max-neighbor
                  :reader max-neighbor)
   (%sample-size :initarg :sample-size
                 :initform nil
                 :reader sample-size)
   (%use-accuracy :initarg :use-accuracy
                  :reader use-accuracy)
   (%use-cluster-size :initarg :use-cluster-size
                      :reader use-cluster-size))
  (:default-initargs
   :use-cluster-size nil
   :use-accuracy nil))
