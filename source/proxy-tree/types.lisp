(cl:in-package #:sl.proxy-tree)


(defclass proxy-tree (sl.mp:fundamental-model-parameters)
  ((%inner :initarg :inner
           :reader inner)))


(defclass honest-tree (proxy-tree)
  ())
