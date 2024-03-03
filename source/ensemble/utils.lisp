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
           (tree-parameters (tree-parameters parameters))
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
               model)))
           (number-of-samples (length samples)))
      (~> (data-points-sampler parameters)
          (data-points-samples state number-of-samples)
          (replace samples _))
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


(defun treep (tree)
  (and (not (null tree))
       (~> tree sl.tp:root sl.tp:treep)))


(defun regression-errors (result prev-trees target-data assigned-leafs parallel)
  (sl.data:data-matrix-map-into result
                                (lambda (index data)
                                  (declare (type fixnum index))
                                  (iterate outer
                                    (declare (type vector leafs)
                                             (type vector assigned-leafs)
                                             (type fixnum index)
                                             (ignorable tree))
                                    (with leafs = (aref assigned-leafs index))
                                    (for i from (~> leafs length 1-) downto 0)
                                    (for tree in-vector prev-trees)
                                    (for leaf = (aref leafs i))
                                    (if (vectorp leaf)
                                        (iterate outer
                                          (for l in-vector leaf)
                                          (for predictions = (sl.tp:predictions l))
                                          (iterate
                                            (declare (type fixnum i))
                                            (for i from 0
                                                 below (array-dimension predictions 1))
                                            (in outer
                                                (summing (abs (- (aref predictions 0 i)
                                                                 (aref data index i)))))))
                                        (bind ((predictions (sl.tp:predictions leaf)))
                                          (iterate
                                            (declare (type fixnum i))
                                            (for i from 0
                                                 below (array-dimension predictions 1))
                                            (in outer
                                                (summing (abs (- (aref predictions 0 i)
                                                                 (aref data index i))))))))))
                                        target-data
                                        parallel))
