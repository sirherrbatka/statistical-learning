(cl:in-package #:cl-user)


(defpackage #:statistical-learning.proxy-tree
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.proxy-tree #:sl.pt)
  (:export
   #:honest
   #:honest-tree
   #:causal-tree
   #:causal
   #:tree-proxy
   #:capture-input
   #:captured-input
   #:triplet
   #:inner))
