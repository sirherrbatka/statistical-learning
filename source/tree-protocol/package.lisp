(cl:in-package #:cl-user)


(defpackage #:statistical-learning.tree-protocol
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:statistical-learning.tp #:sl.tp)
  (:intern sl.opt:left sl.opt:right)
  (:export
   #:attribute-indexes
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
   #:fundamental-splitter
   #:fundamental-tree-node
   #:fundamental-tree-training-parameters
   #:indexes
   #:initialize-leaf
   #:leaf-for
   #:leafp
   #:leafs-for
   #:left-node
   #:loss
   #:make-leaf
   #:make-leaf*
   #:make-node
   #:maximal-depth
   #:minimal-difference
   #:minimal-size
   #:parallel
   #:pick-split
   #:pick-split*
   #:point
   #:predictions
   #:random-attribute-splitter
   #:right-node
   #:root
   #:split
   #:split*
   #:split-training-state
   #:split-training-state*
   #:split-training-state-info
   #:splitter
   #:fill-split-vector*
   #:pick-split*
   #:requires-split-p
   #:standard-leaf-node
   #:standard-tree-training-parameters
   #:sums
   #:support
   #:training-state-clone
   #:tree-model
   #:tree-training-state
   #:treep
   #:trials-count
   #:visit-nodes))
