(cl:in-package #:statistical-learning.model-protocol)


(defclass fundamental-model ()
  ((%parameters :initarg :parameters
                :reader parameters)))


(defclass fundamental-model-parameters ()
  ())


(defclass fundamental-training-state ()
  ((%training-parameters :initarg :training-parameters
                         :reader training-parameters)
   (%target-data :initarg :target-data
                 :reader target-data)
   (%weights :initarg :weights
             :reader weights)
   (%training-data :initarg :training-data
                   :reader training-data))
  (:default-initargs :weights nil))
