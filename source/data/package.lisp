(cl:in-package #:cl-user)


(defpackage #:cl-grf.data
  (:use #:cl #:cl-grf.aux-package)
  (:export
   #:attributes-count
   #:bind-data-matrix-dimensions
   #:data-matrix
   #:data-matrix-dimensions
   #:mref
   #:iota-vector
   #:reshuffle
   #:select-random-indexes
   #:selecting-random-indexes
   #:make-data-matrix
   #:sample
   #:data-points-count))
