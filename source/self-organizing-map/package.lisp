(cl:in-package #:cl-user)


(defpackage #:statistical-learning.self-organizing-map
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.som)
  (:export
   #:self-organizing-map
   #:find-best-matching-unit
   #:<euclid-matching-unit-selector>
   #:<linear-decay>
   #:<hill-decay>
   #:alpha
   #:sigma
   #:unit-at
   #:decay))
