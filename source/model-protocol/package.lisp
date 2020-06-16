(cl:in-package #:cl-user)


(defpackage #:statistical-learning.model-protocol
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.mp #:sl.mp)
  (:export
   #:fundamental-model
   #:fundamental-model-parameters
   #:make-model
   #:parameters
   #:predict
   ))
