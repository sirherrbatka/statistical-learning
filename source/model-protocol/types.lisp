(cl:in-package #:statistical-learning.model-protocol)


(defclass fundamental-model (sl.common:proxy-enabled)
  ((%parameters :initarg :parameters
                :reader parameters)))


(defclass fundamental-model-parameters (sl.common:proxy-enabled)
  ())


(defclass fundamental-training-state ()
  ((%training-parameters :initarg :training-parameters
                         :accessor training-parameters)
   (%cached :initarg :cached
            :reader cached)
   (%model-cached :initarg :model-cached
                  :reader model-cached))
  (:default-initargs :cached (make-hash-table)
                     :model-cached (make-hash-table)))


(defclass supervised-model (fundamental-model)
  ())
