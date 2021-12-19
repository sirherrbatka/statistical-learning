(cl:in-package #:statistical-learning.ensemble)


(defgeneric trees (ensemble))
(defgeneric trees-count (ensemble-parameters))
(defgeneric tree-attributes-count (ensemble-parameters))
(defgeneric parallel (parameters))
(defgeneric tree-parameters (parameters))
(defgeneric tree-batch-size (parameters))
(defgeneric update-weights (calculator
                            tree-parameters
                            ensemble-state
                            ensemble-model))
(defgeneric leafs (ensemble data &optional parallel))
(defgeneric assign-leafs (state model))
(sl.common:defgeneric/proxy make-tree-training-state
    ((ensemble-parameters)
     (tree-parameters)
     ensemble-state
     attributes
     data-points
     initargs))
(defgeneric data-points-samples (sampler state count))
(sl.common:defgeneric/proxy after-tree-fitting
    ((ensemble-parameters)
     (tree-parametres)
     ensemble-state))
(defgeneric indexes (ensemble-state))
(defgeneric assigned-leafs (ensemble-state))
(defgeneric (setf assigned-leafs) (new-value ensemble-state))
(defgeneric leafs-assigned-p (ensemble-state))
(defgeneric (setf leafs-assigned-p) (new-value ensemble-state))
(defgeneric trees-view (ensemble-state))
(defgeneric (setf trees-view) (new-value ensemble-state))
(defgeneric samples-view (ensemble-state))
(defgeneric (setf samples-view) (new-value ensemble-state))
(defgeneric make-weights-calculator-state (weights-calculator ensemble-state))
(defgeneric make-data-point-sampler-state (data-points-sampler ensemble-state))
(defgeneric prune-trees (algorithm ensemble train-data target-data))
