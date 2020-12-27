(cl:in-package #:cl-user)


(defpackage #:statistical-learning.isolation-forest
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.if #:sl.if)
  (:export
   #:c-factor
   #:c
   #:make-normals
   #:calculate-mins
   #:calculate-maxs
   #:isolation
   #:isolation-splitter))
