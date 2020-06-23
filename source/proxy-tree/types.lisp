(cl:in-package #:sl.proxy-tree)


(defclass proxy-tree (sl.tp:fundamental-tree-training-parameters)
  ((%inner :initarg :inner
           :reader inner)))


(defclass proxy-state (sl.mp:fundamental-training-state)
  ((%inner :initarg :inner
           :reader inner)))
