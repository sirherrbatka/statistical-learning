(cl:in-package #:statistical-learning.model-protocol)


(defclass fundamental-model (sl.common:proxy-enabled)
  ((%parameters :initarg :parameters
                :reader parameters)))


(defclass fundamental-model-parameters (sl.common:proxy-enabled)
  ())


(defclass fundamental-training-state ()
  ((%training-parameters :initarg :training-parameters
                         :accessor training-parameters)))


(defclass supervised-model (fundamental-model)
  ())
