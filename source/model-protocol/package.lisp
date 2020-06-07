(cl:in-package #:cl-user)


(defpackage #:cl-grf.model-protocol
  (:use #:cl #:cl-grf.aux-package)
  (:nicknames #:cl-grf.mp)
  (:export
   #:make-model
   #:parameters
   #:fundamental-model
   #:fundamental-model-parameters
   #:predict))
