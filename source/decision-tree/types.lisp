(cl:in-package #:statistical-learning.decision-tree)


(defclass fundamental-decision-tree-parameters
    (sl.tp:standard-tree-training-parameters)
  ((%optimized-function :initarg :optimized-function
                        :reader sl.opt:optimized-function
                        :reader optimized-function)))


(defclass classification (sl.perf:classification sl.tp:supervised fundamental-decision-tree-parameters)
  ())


(defclass regression (sl.perf:regression sl.tp:supervised fundamental-decision-tree-parameters)
  ())
