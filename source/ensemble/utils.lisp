(cl:in-package #:statistical-learning.ensemble)


(defun weighted-sample (sample-size distribution)
  (map-into (make-array sample-size :element-type 'fixnum)
            distribution))


(defun bootstrap-sample (tree-sample-size data-points-count distribution)
  (if (null distribution)
      (sl.data:select-random-indexes tree-sample-size
                                     data-points-count)
      (weighted-sample tree-sample-size distribution)))


(defun sample-to-dict)


(defun fit-tree-batch (parameters trees all-attributes state sampling-weights samples)
  (bind ((parallel (parallel parameters))
         (tree-sample-rate (tree-sample-rate parameters))
         (data-points-count (~> state sl.mp:training-data sl.data:data-points-count))
         (tree-sample-size (floor (* tree-sample-rate data-points-count)))
         (distribution (if (null sampling-weights)
                           nil
                           (sl.random:discrete-distribution sampling-weights)))
         ((:flet make-model (attributes sample))
          (bind ((sub-state (sl.mp:sample-training-state state
                                                         :data-points sample
                                                         :train-attributes attributes)))
            (sl.mp:make-model* (sl.mp:training-parameters sub-state) sub-state))))
    (map-into samples (curry #'bootstrap-sample
                             tree-sample-size data-points-count distribution))
    (funcall (if parallel #'lparallel:pmap-into #'map-into)
             trees
             #'make-model
             all-attributes
             samples)))


(defun contribute-trees (tree-parameters trees data parallel &optional state)
  (iterate
    (for tree in-vector trees)
    (setf state (statistical-learning.tp:contribute-predictions* tree-parameters
                                                   tree
                                                   data
                                                   state
                                                   parallel))
    (finally (return state))))


(defun trees-predict (tree-parameters trees data parallel &optional state)
  (let ((state (contribute-trees tree-parameters trees data parallel state)))
    (values (statistical-learning.tp:extract-predictions state)
            state)))
