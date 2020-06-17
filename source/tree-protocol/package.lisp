(cl:in-package #:cl-user)


(defpackage #:statistical-learning.tree-protocol
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.tp #:sl.tp)
  (:intern sl.opt:left sl.opt:right)
  (:export
   #:attribute
   #:attribute-indexes
   #:attribute-value
   #:calculate-loss*
   #:contribute-predictions
   #:contribute-predictions*
   #:contributed-predictions
   #:contributions-count
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
   #:indexes
   #:leaf-for
   #:leafp
   #:leafs-for
   #:left-node
   #:loss
   #:make-leaf
   #:make-leaf*
   #:initialize-leaf
   #:make-training-state
   #:split-training-state*
   #:split-training-state
   #:make-node
   #:maximal-depth
   #:minimal-difference
   #:minimal-size
   #:parallel
   #:predictions
   #:right-node
   #:root
   #:split
   #:split*
   #:sums
   #:support
   #:target-data
   #:training-data
   #:training-parameters
   #:training-state-clone
   #:tree-model
   #:treep
   #:trials-count
   #:visit-nodes
   #:weights
   ))
