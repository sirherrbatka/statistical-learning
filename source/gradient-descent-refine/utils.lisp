(cl:in-package #:statistical-learning.gradient-descent-refine)


(defun refine-implementation (algorithm
                              ensemble
                              train-data
                              target-data)
  (bind ((parallel (parallel algorithm))
         (epochs (epochs algorithm))
         (data-points-count (sl.data:data-points-count train-data))
         (trees (sl.ensemble:trees ensemble))
         (total-weight (reduce #'+ trees :key #'sl.tp:weight))
         (leafs (the sl.data:universal-data-matrix (sl.ensemble:leafs ensemble
                                                                      train-data
                                                                      parallel)))
         (indexes (sl.data:iota-vector data-points-count))
         (leaf-locks (when parallel
                       (~> indexes
                           (cl-ds.alg:flatten-lists :key (lambda (index)
                                                           (let ((result (sl.data:mref leafs index 0)))
                                                             (if (vectorp result)
                                                                 (coerce result 'list)
                                                                 result))))
                           (cl-ds.alg:group-by :test 'eq)
                           (cl-ds.alg:to-list :after (lambda (list) (declare (ignore list))
                                                       (bt:make-lock))))))
         (initial-shrinkage (shrinkage algorithm))
         (shrinkage initial-shrinkage)
         (sample-size (sample-size algorithm))
         ((:flet adjust-predictions (data-point column difference))
          (iterate
            (with leaf-vector = (sl.data:mref leafs data-point 0))
            (with number-of-trees = (length trees))
            (for i from 0 below number-of-trees)
            (for leaf = (aref leaf-vector i))
            (if (vectorp leaf)
                (iterate
                  (with length = (length leaf))
                  (for ii from 0 below length)
                  (for l = (aref leaf ii))
                  (for predictions = (sl.tp:predictions l))
                  (incf (sl.data:mref predictions 0 column)
                        (/ (* shrinkage difference)
                           length)))
                (let ((predictions (sl.tp:predictions leaf)))
                  (incf (aref predictions 0 column)
                        (* shrinkage difference))))))
         ((:flet prediction (data-point &aux (leaf-vector (sl.data:mref leafs data-point 0))))
          (iterate
            (with number-of-trees = (length trees))
            (with sums = (make-array number-of-trees
                                     :element-type 'double-float
                                     :initial-element 0.0d0))
            (for i from 0 below number-of-trees)
            (for tree = (aref trees i))
            (for weight = (sl.tp:weight tree))
            (for leaf = (aref leaf-vector i))
            (if (vectorp leaf)
                (iterate
                  (with length = (length leaf))
                  (for ii from 0 below length)
                  (for l = (aref leaf ii))
                  (for predictions = (sl.tp:predictions l))
                  (iterate
                    (for ii from 0 below (array-dimension predictions 1))
                    (for pred = (aref predictions 0 ii))
                    (incf (aref sums ii) (/ (* weight pred) length))))
                (let ((predictions (sl.tp:predictions leaf)))
                  (iterate
                    (for ii from 0 below (array-dimension predictions 1))
                    (for pred = (aref predictions 0 ii))
                    (incf (aref sums ii) (* weight pred)))))
            (finally
             (return (cl-ds.utils:transform (rcurry #'/ total-weight)
                                                    sums))))))
    (iterate
      (with batch-size = (if sample-size
                             (min sample-size
                                  (length indexes))
                             (length indexes)))
      (for epoch from epochs downto 1)
      (setf shrinkage (* initial-shrinkage (/ epoch epochs)))
      (sl.data:reshuffle indexes)
      (for batch = (make-array batch-size :element-type 'fixnum
                                          :displaced-to indexes))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               (lambda (data-point &aux (leaf-vector (sl.data:mref leafs data-point 0)))
                 (when parallel
                   (iterate
                     (for leaf in-vector leaf-vector)
                     (if (vectorp leaf)
                         (iterate
                           (for l in-vector leaf)
                           (bt:release-lock (cl-ds:at leaf-locks l)))
                         (bt:release-lock (cl-ds:at leaf-locks leaf)))))
                 (iterate
                   (with prediction = (prediction data-point))
                   (for column from 0 below (sl.data:attributes-count target-data))
                   (for difference = (- (sl.data:mref target-data data-point column)
                                        (aref prediction column)))
                   (adjust-predictions data-point column difference))
                 (when parallel
                   (iterate
                     (for leaf in-vector leaf-vector)
                     (if (vectorp leaf)
                         (iterate
                           (for l in-vector leaf)
                           (bt:release-lock (cl-ds:at leaf-locks l)))
                         (bt:release-lock (cl-ds:at leaf-locks leaf))))))
               batch)
      (finally (return ensemble)))))
