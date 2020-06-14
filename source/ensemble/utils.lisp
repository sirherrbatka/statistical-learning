(cl:in-package #:statistical-learning.ensemble)


(-> fit-tree-batch (vector
                    vector
                    t
                    statistical-learning.data:data-matrix
                    statistical-learning.data:data-matrix
                    (or null statistical-learning.data:data-matrix))
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
         (data-points-count (statistical-learning.data:data-points-count train-data))
         (tree-sample-size (ceiling (* tree-sample-rate data-points-count)))
         (distribution (if (null weights)
                           nil
                           (statistical-learning.random:discrete-distribution weights))))
    (funcall (if parallel #'lparallel:pmap-into #'map-into)
             trees
             (lambda (attributes)
               (bind ((sample (if (null distribution)
                                  (statistical-learning.data:select-random-indexes tree-sample-size
                                                                     data-points-count)
                                  (map-into (make-array tree-sample-size
                                                        :element-type 'fixnum)
                                            distribution)))
                      (train (statistical-learning.data:sample
                              train-data
                              :attributes attributes
                              :data-points sample))
                      (target (statistical-learning.data:sample
                               target-data
                               :data-points sample)))
                 (assert (= (statistical-learning.data:attributes-count train)
                            (length attributes)))
                 (assert (= (statistical-learning.data:data-points-count target)
                            (length sample)))
                 (statistical-learning.mp:make-model tree-parameters
                                       train
                                       target
                                       :attributes attributes)))
             all-attributes)))


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
