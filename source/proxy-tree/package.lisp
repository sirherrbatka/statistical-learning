(cl:in-package #:cl-user)


(defpackage #:statistical-learning.proxy-tree
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.proxy-tree #:sl.pt)
  (:export
   #:honest
   #:causal
   #:indexing
   #:index
   #:inner
   #:forward-call
   #:proxy-tree
   ))
