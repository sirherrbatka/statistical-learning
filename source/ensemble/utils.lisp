(cl:in-package #:statistical-learning.ensemble)


(defun weighted-sample (sample-size distribution)
  (map-into (make-array sample-size :element-type 'fixnum)
            distribution))


(defun fit-tree-batch (initargs state)
  (cl-ds.utils:with-rebind (cl-progress-bar:*progress-bar*)
    (bind ((parameters (sl.mp:parameters state))
           (parallel (parallel parameters))
           (trees (trees-view state))
           (samples (samples-view state))
           (all-attributes (attributes-view state))
           (tree-sample-rate (tree-sample-rate parameters))
           (tree-parameters (tree-parameters parameters))
           (train-data (sl.mp:train-data state))
           (data-points-count (sl.data:data-points-count train-data))
           (tree-sample-size (ceiling (* tree-sample-rate
                                         data-points-count)))
           (complete-initargs (append initargs (all-args state)))
           ((:flet make-model (attributes sample))
            (cl-ds.utils:rebind
             (bind ((*state* state)
                    (sub-state (make-tree-training-state
                                parameters
                                tree-parameters
                                state
                                attributes
                                (sort sample #'<)
                                complete-initargs))
                    (model (sl.mp:make-model* tree-parameters
                                              sub-state)))
               (cl-progress-bar:update 1)
               model))))
      (replace samples (data-point-samples (data-points-sampler parameters)
                                           (length samples)
                                           state
                                           tree-sample-size
                                           data-points-count))
      (funcall (if parallel #'lparallel:pmap-into #'map-into)
               trees
               #'make-model
               all-attributes
               samples))))


(defun contribute-trees (ensemble tree-parameters trees data parallel
                         &optional state)
  (iterate
    (for tree in-vector trees)
    (setf state (sl.tp:contribute-predictions* tree-parameters
                                               tree
                                               data
                                               state
                                               ensemble
                                               parallel))
    (finally (return state))))


(defun trees-predict (ensemble tree-parameters trees data parallel
                      &optional state)
  (let ((state (contribute-trees ensemble tree-parameters trees
                                 data parallel state)))
    (values (statistical-learning.tp:extract-predictions state)
            state)))


(defun treep (tree)
  (and (not (null tree))
       (~> tree sl.tp:root sl.tp:treep)))
