(cl:in-package #:statistical-learning.decision-tree)


(defclass fundamental-decision-tree-parameters
    (sl.tp:standard-tree-training-parameters)
  ((%optimized-function :initarg :optimized-function
                        :reader optimized-function)))


(defclass classification (sl.perf:classification
                          fundamental-decision-tree-parameters)
  ())


(defclass regression (sl.perf:regression
                      fundamental-decision-tree-parameters)
  ())
