(cl:in-package #:cl-user)


(defpackage #:cl-grf.tree-protocol
  (:use #:cl #:cl-grf.aux-package)
  (:nicknames #:cl-grf.tp)
  (:export
   #:depth
   #:force-tree
   #:fundamental-leaf-node
   #:fundamental-node
   #:fundamental-tree-training-parameters
   #:fundamental-training-state
   #:fundamental-tree-node
   #:force-tree*
   #:leafp
   #:make-node
   #:training-data
   #:trials-count
   #:training-parameters
   #:attribute
   #:target-data
   #:parallel
   #:maximal-depth
   #:attribute-value
   #:left-node
   #:right-node
   #:training-state-clone
   #:split
   #:split*
   #:attribute-indexes
   #:leaf-for
   #:leafs-for
   #:make-leaf
   #:make-leaf*
   #:minimal-size
   #:treep
   ))
