(cl:in-package #:sl.proxy-tree)


(defclass proxy-state (sl.mp:fundamental-training-state)
  ((%inner :initarg :inner
           :reader inner)))


(defclass tree-proxy (sl.common:lifting-proxy)
  ())

