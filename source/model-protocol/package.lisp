(cl:in-package #:cl-user)


(defpackage #:statistical-learning.model-protocol
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.mp)
  (:export
   #:make-model
   #:parameters
   #:fundamental-model
   #:fundamental-model-parameters
   #:predict))
