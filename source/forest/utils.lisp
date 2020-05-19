(cl:in-package #:cl-grf.forest)


(defun total-support (leafs index)
  (iterate
    (for l in-vector leafs)
    (sum (cl-grf.alg:support (aref l index)))))


(defun classification-sums-from-leafs* (leafs parallel &optional results)
  (bind ((length (~> leafs first-elt length))
         (effective-results (or results
                                (make-array length
                                            :initial-element nil)))
         ((:flet prediction (i))
          (declare (type fixnum i))
          (iterate
            (with result = (aref effective-results i))
            (for leaf-group in-vector leafs)
            (for leaf = (aref leaf-group i))
            (for predictions = (cl-grf.alg:predictions leaf))
            (for data-points-count = (cl-grf.data:data-points-count
                                      predictions))
            (for attributes-count = (cl-grf.data:attributes-count predictions))
            (ensure result
              (cl-grf.data:make-data-matrix data-points-count
                                            attributes-count))
            (for support = (cl-grf.alg:support leaf))
            (iterate
              (for k from 0 below (array-total-size predictions))
              (incf (row-major-aref result k)
                    (/ (row-major-aref predictions k)
                       support)))
            (finally (return result)))))
    (funcall (if parallel #'lparallel:pmap-into #'map-into)
             effective-results
             #'prediction
             (cl-grf.data:iota-vector length))
    effective-results))


(defun classification-predictions-from-sums* (sums trees-count results)
  (iterate
    (with length = (length sums))
    (for i from 0 below length)
    (for sum = (aref sums i))
    (for result = (aref results i))
    (iterate
      (for k from 0 below (array-total-size result))
      (setf (row-major-aref result k)
            (/ (row-major-aref sum k)
               trees-count)))
    (finally (return results))))


(defun classification-predictions-from-leafs* (leafs &optional parallel)
  (let ((sums (classification-sums-from-leafs* leafs parallel)))
    (classification-predictions-from-sums* sums
                                           (length leafs)
                                           sums)))


(defun leafs-for* (trees attributes data &optional parallel)

  (funcall (if parallel
               #'lparallel:pmap
               #'map)
           'vector
           (lambda (tree features)
             (~>> (cl-grf.data:sample data :attributes features)
                  (cl-grf.tp:leafs-for tree)))
            trees
            attributes))


(-> calculate-weights (simple-vector
                       cl-grf.data:data-matrix fixnum
                       &optional (or null (cl-grf.data:data-matrix)))
    cl-grf.data:data-matrix)
(defun calculate-weights (predictions target base &optional result)
  (declare (optimize (speed 3) (safety 0)))
  (bind ((length (length predictions)))
    (ensure result
      (cl-grf.data:make-data-matrix length 1))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below length)
      (for expected = (cl-grf.data:mref target i 0))
      (for prediction = (cl-grf.data:mref (aref predictions i)
                                          0
                                          (truncate expected)))
      (setf (cl-grf.data:mref result i 0) (- (log (max prediction double-float-epsilon)
                                                  base))))
    result))


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
                              :data-points sample
                              :attributes attributes))
                      (target (cl-grf.data:sample
                               target-data
                               :data-points sample)))
                 (cl-grf.mp:make-model tree-parameters
                                       train
                                       target)))
             (array-view all-attributes))))
