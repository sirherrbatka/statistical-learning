(cl:in-package #:statistical-learning.tree-protocol)

(sl.common:defgeneric/proxy contribute-predictions*
    ((parameters)
     model data state context parallel
     &optional leaf-key))

(sl.common:defgeneric/proxy split-training-state-info
    (splitter
     (parameters)
     state split-array
     position size point)
  (:method-combination append :most-specific-last))

(sl.common:defgeneric/proxy leaf-for ((splitter) node data index context))

(defgeneric split* (training-parameters training-state))

(sl.common:defgeneric/proxy split-using-splitter
    ((splitter) parameters state))

(sl.common:defgeneric/proxy split-training-state*
    ((parameters) state split-array position size initargs point))

(sl.common:defgeneric/proxy make-leaf* ((training-parameters) state))

(sl.common:defgeneric/proxy requires-split-p
    (splitter (parameters) training-state)
  (:method-combination and))

(sl.common:defgeneric/proxy initialize-leaf ((training-parameters)
                                             training-state
                                             leaf))

(sl.common:defgeneric/proxy extract-predictions* ((parameters)
                                                  state))

(sl.common:defgeneric/proxy fill-split-vector*
    ((splitter) parameters state point split-vector))

(sl.common:defgeneric/proxy pick-split*
    ((splitter) parameters state))

(sl.common:defgeneric/proxy calculate-loss*
    ((parameters) state split-array left-length right-length))

(sl.common:defgeneric/proxy split-result-accepted-p
    ((parameters)
     state
     result))

(sl.common:defgeneric/proxy split-result-improved-p
    ((parameters)
     state
     new-result
     old-result))

(defgeneric root (model))
(defgeneric treep (node))
(defgeneric leafp (node))
(defgeneric maximal-depth (training-parameters))
(defgeneric depth (state))
(defgeneric make-node (node-class &rest arguments))
(defgeneric trials-count (training-parameters))
(defgeneric force-tree* (tree))
(defgeneric left-node (tree))
(defgeneric (setf left-node) (new-value tree))
(defgeneric right-node (tree))
(defgeneric (setf right-node) (new-value tree))
(defgeneric minimal-size (training-parameters))
(defgeneric attribute (tree-node))
(defgeneric (setf attribute) (new-value tree-node))
(defgeneric target-data (training-state))
(defgeneric (setf target-data) (new-value training-state))
(defgeneric parallel (training-parameters))
(defgeneric attribute-value (tree-node))
(defgeneric (setf attribute-value) (new-value tree-node))
(defgeneric attribute-indexes (training-state))
(defgeneric (setf attribute-indexes) (new-value training-state))
(defgeneric loss (state))
(defgeneric weights (state))
(defgeneric support (node))
(defgeneric (setf support) (new-value node))
(defgeneric predictions (node))
(defgeneric (setf predictions) (new-value node))
(defgeneric sums (predictions))
(defgeneric (setf sums) (new-value predictions))
(defgeneric indexes (predictions))
(defgeneric contributions-count (predictions))
(defgeneric training-parameters (predictions))
(defgeneric split-point (tree-training-state))
(defgeneric (setf split-point) (new-value tree-training-state))
(defgeneric distance-function (splitter))
(defgeneric weight (model))
(defgeneric tree-parameters (parent-parameters))
(defgeneric parent-state (tree-training-state))
