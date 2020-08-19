(cl:in-package #:cl-user)


(defpackage #:statistical-learning.self-organizing-map
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.som)
  (:export
   #:self-organizing-map
   #:<linear-decay>
   #:<hill-decay>
   #:alpha
   #:sigma
   #:unit-at
   #:decay))
