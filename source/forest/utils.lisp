(cl:in-package #:cl-grf.forest)


(defun total-support (leafs index)
  (iterate
    (for l in-vector leafs)
    (sum (cl-grf.alg:support (aref l index)))))


(defgeneric calculate-weights (parameters predictions target base &optional result))


(defmethod calculate-weights ((parameters classification-random-forest-parameters)
                              predictions target base &optional result)
  (declare (optimize (speed 3) (safety 0))
           (type cl-grf.data:data-matrix target predictions)
           (type fixnum base))
  (cl-grf.data:bind-data-matrix-dimensions ((length classes predictions)
                                            (data-points attributes target))
    (ensure result
      (cl-grf.data:make-data-matrix length 1))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below length)
      (for expected = (cl-grf.data:mref target i 0))
      (for prediction = (cl-grf.data:mref predictions
                                          i
                                          (truncate expected)))
      (setf (cl-grf.data:mref result i 0) (- (log (max prediction double-float-epsilon)
                                                  base))))
    result))


(defmethod calculate-weights ((parameters regression-random-forest-parameters)
                              predictions target base &optional result)
  (declare (optimize (speed 3) (safety 2))
           (type cl-grf.data:data-matrix target predictions)
           (type fixnum base))
  (cl-grf.data:bind-data-matrix-dimensions ((length classes predictions)
                                            (data-points attributes target))
    (ensure result
      (cl-grf.data:make-data-matrix length 1))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below length)
      (for expected = (cl-grf.data:mref target i 0))
      (for prediction = (cl-grf.data:mref predictions i 0))
      (for error = (- expected prediction))
      (setf (cl-grf.data:mref result i 0) (* error error))))
  result)


(-> fit-tree-batch (simple-vector
                    simple-vector
                    fixnum
                    t
                    cl-grf.data:data-matrix
                    cl-grf.data:data-matrix
                    &optional (or null cl-grf.data:data-matrix))
    t)
(defun fit-tree-batch (trees
                       all-attributes
                       fill-pointer
                       parameters
                       train-data
                       target-data
                       &optional weights)
  (bind ((tree-parameters (tree-parameters parameters))
         (tree-batch-size (tree-batch-size parameters))
         (parallel (parallel parameters))
         (tree-sample-rate (tree-sample-rate parameters))
         (data-points-count (cl-grf.data:data-points-count train-data))
         (tree-sample-size (ceiling (* tree-sample-rate data-points-count)))
         (tree-maximum-count (length trees))
         (real-batch-size (min tree-batch-size
                               (- tree-maximum-count fill-pointer)))
         (distribution (if (null weights)
                           nil
                           (cl-grf.random:discrete-distribution weights)))
         ((:flet array-view (array))
          (make-array real-batch-size
                      :displaced-to array
                      :displaced-index-offset fill-pointer)))
    (funcall (if parallel #'lparallel:pmap-into #'map-into)
             (array-view trees)
             (lambda (attributes)
               (bind ((sample (if (null distribution)
                                  (cl-grf.data:select-random-indexes tree-sample-size
                                                                     data-points-count)
                                  (map-into (make-array tree-sample-size
                                                        :element-type 'fixnum)
                                            distribution)))
                      (train (cl-grf.data:sample
                              train-data
                              :data-points sample))
                      (target (cl-grf.data:sample
                               target-data
                               :data-points sample)))
                 (cl-grf.mp:make-model tree-parameters
                                       train
                                       target
                                       :attributes attributes)))
             (array-view all-attributes))))


(defun trees-predict (tree-parameters trees data parallel &optional state)
  (declare (optimize (debug 3)))
  (iterate
    (for tree in-vector trees)
    (setf state (cl-grf.tp:contribute-predictions tree-parameters
                                                  tree
                                                  data
                                                  state
                                                  parallel))
    (finally
     (let ((result (cl-grf.tp:extract-predictions state)))
       (return (values result state))))))
