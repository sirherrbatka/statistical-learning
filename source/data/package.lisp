(cl:in-package #:cl-user)


(defpackage #:cl-grf.data
  (:use #:cl #:cl-grf.aux-package)
  (:export
   #:attributes-count
   #:bind-data-matrix-dimensions
   #:data-matrix
   #:data-matrix-dimensions
   #:mref
   #:map-data-matrix
   #:make-data-matrix-like
   #:reduce-data-points
   #:iota-vector
   #:reshuffle
   #:select-random-indexes
   #:selecting-random-attributes
   #:selecting-random-indexes
   #:cross-validation-folds
   #:check-data-points
   #:make-data-matrix
   #:sample
   #:data-points-count))
