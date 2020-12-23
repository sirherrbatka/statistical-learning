(cl:in-package #:cl-user)


(defpackage #:statistical-learning.isolation-forest
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.if #:sl.if)
  (:export
   #:c-factor
   #:c
   #:mins
   #:maxs
   #:global-min
   #:global-max
   #:isolation
   #:isolation-splitter))
