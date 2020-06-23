(cl:in-package #:statistical-learning.model-protocol)


(defclass fundamental-model ()
  ((%parameters :initarg :parameters
                :reader parameters)))


(defclass fundamental-model-parameters ()
  ())


(defclass fundamental-training-state ()
  ((%training-parameters :initarg :training-parameters
                         :accessor training-parameters)))
