(cl:in-package #:sl.ct)


(defclass causal (sl.tp:fundamental-tree-training-parameters)
  ((%minimal-treatment-size :reader minimal-treatment-size
                            :initarg :minimal-treatment-size)
   (%minimal-no-treatment-size :reader minimal-no-treatment-size
                               :initarg :minimal-no-treatment-size)))


(defclass causal-classificaton (causal)
  ((%number-of-classes :initarg :number-of-classes
                       :reader number-of-classes)))


(defclass causal-regression (causal)
  ())


(defclass causal-tree-training-state (sl.tp:tree-training-state)
  ((%treatment :initarg :treatment
               :reader treatment)))
