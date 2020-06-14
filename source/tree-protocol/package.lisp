(cl:in-package #:cl-user)


(defpackage #:cl-grf.tree-protocol
  (:use #:cl #:cl-grf.aux-package)
  (:nicknames #:cl-grf.tp)
  (:export
   #:attribute
   #:attribute-indexes
   #:attribute-value
   #:contribute-predictions
   #:contribute-predictions*
   #:contributed-predictions
   #:depth
   #:extract-predictions
   #:extract-predictions*
   #:force-tree
   #:force-tree*
   #:fundamental-leaf-node
   #:fundamental-node
   #:fundamental-training-state
   #:fundamental-tree-node
   #:fundamental-tree-training-parameters
   #:leaf-for
   #:leafp
   #:leafs-for
   #:left-node
   #:make-leaf
   #:make-leaf*
   #:make-node
   #:maximal-depth
   #:minimal-size
   #:parallel
   #:right-node
   #:root
   #:split
   #:split*
   #:target-data
   #:training-data
   #:training-parameters
   #:training-state-clone
   #:tree-model
   #:treep
   #:trials-count
   #:visit-nodes))
