(cl:in-package #:cl-grf.model-protocol)


(defclass fundamental-model ()
  ((%parameters :initarg :parameters
                :reader parameters)))


(defclass fundamental-model-parameters ()
  ())
