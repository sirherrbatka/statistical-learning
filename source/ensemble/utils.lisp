(cl:in-package #:cl-grf.ensemble)


(-> fit-tree-batch (vector
                    vector
                    t
                    cl-grf.data:data-matrix
                    cl-grf.data:data-matrix
                    (or null cl-grf.data:data-matrix))
    t)
(defun fit-tree-batch (trees
                       all-attributes
                       parameters
                       train-data
                       target-data
                       weights)
  (bind ((tree-parameters (tree-parameters parameters))
         (parallel (parallel parameters))
         (tree-sample-rate (tree-sample-rate parameters))
         (data-points-count (cl-grf.data:data-points-count train-data))
         (tree-sample-size (ceiling (* tree-sample-rate data-points-count)))
         (distribution (if (null weights)
                           nil
                           (cl-grf.random:discrete-distribution weights))))
    (funcall (if parallel #'lparallel:pmap-into #'map-into)
             trees
             (lambda (attributes)
               (bind ((sample (if (null distribution)
                                  (cl-grf.data:select-random-indexes tree-sample-size
                                                                     data-points-count)
                                  (map-into (make-array tree-sample-size
                                                        :element-type 'fixnum)
                                            distribution)))
                      (train (cl-grf.data:sample
                              train-data
                              :attributes attributes
                              :data-points sample))
                      (target (cl-grf.data:sample
                               target-data
                               :data-points sample)))
                 (assert (= (cl-grf.data:attributes-count train)
                            (length attributes)))
                 (assert (= (cl-grf.data:data-points-count target)
                            (length sample)))
                 (cl-grf.mp:make-model tree-parameters
                                       train
                                       target
                                       :attributes attributes)))
             all-attributes)))


(defun contribute-trees (tree-parameters trees data parallel &optional state)
  (iterate
    (for tree in-vector trees)
    (setf state (cl-grf.tp:contribute-predictions* tree-parameters
                                                   tree
                                                   data
                                                   state
                                                   parallel))
    (finally (return state))))


(defun trees-predict (tree-parameters trees data parallel &optional state)
  (let ((state (contribute-trees tree-parameters trees data parallel state)))
    (values (cl-grf.tp:extract-predictions state)
            state)))
