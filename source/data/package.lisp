(cl:in-package #:cl-user)


(defpackage #:cl-grf.data
  (:use #:cl #:cl-grf.aux-package)
  (:export
   #:attributes-count
   #:bind-data-matrix-dimensions
   #:data-matrix
   #:data-matrix-dimensions
   #:mref
   #:make-data-matrix
   #:data-points-count))
