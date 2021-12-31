(cl:in-package #:statistical-learning.gradient-descent-refine)


(defclass parameters (sl.ensemble:fundamental-refinement-algorithm)
  ((%parallel :initarg :parallel
              :reader parallel)
   (%shrinkage :initarg :shrinkage
               :reader shrinkage)
   (%sample-size :initarg :sample-size
                 :reader sample-size)
   (%epochs :initarg :epochs
            :reader epochs))
  (:default-initargs
   :epochs 5
   :sample-size 1000
   :shrinkage 0.1
   :parallel nil))
