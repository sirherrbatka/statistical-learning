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
         (leafs (sl.ensemble:leafs ensemble
                                   train-data
                                   parallel))
         (indexes (sl.data:iota-vector data-points-count))
         (leaf-locks (when parallel
                       (~> indexes
                           (cl-ds.alg:multiplex :key (lambda (index)
                                                       (aref leafs index 0)))
                           (cl-ds.alg:group-by :test 'eq)
                           (cl-ds.alg:to-list :after (lambda (list) (declare (ignore list))
                                                       (bt:make-lock))))))
         (initial-shrinkage (shrinkage algorithm))
         (shrinkage initial-shrinkage)
         (sample-size (sample-size algorithm))
         ((:flet adjust-predictions (data-point column difference))
          (iterate
            (with leaf-vector = (aref leafs data-point 0))
            (with number-of-trees = (length trees))
            (for i from 0 below number-of-trees)
            (for leaf = (aref leaf-vector i))
            (for predictions = (sl.tp:predictions leaf))
            (incf (sl.data:mref predictions 0 column)
                  (* shrinkage difference))))
         ((:flet prediction (data-point &aux (leaf-vector (aref leafs data-point 0))))
          (iterate
            (with number-of-trees = (length trees))
            (with sums = (make-array number-of-trees
                                     :element-type 'double-float
                                     :initial-element 0.0d0))
            (for i from 0 below number-of-trees)
            (for tree = (aref trees i))
            (for weight = (sl.tp:weight tree))
            (for leaf = (aref leaf-vector i))
            (for predictions = (sl.tp:predictions leaf))
            (bt:with-lock-held ((cl-ds:at leaf-locks leaf))
              (iterate
                (for ii from 0 below (sl.data:attributes-count predictions))
                (for pred = (sl.data:mref predictions 0 ii))
                (incf (aref sums ii) (* weight pred))))
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
               (lambda (data-point &aux (leaf-vector (aref leafs data-point 0)))
                 (when parallel
                   (iterate
                     (for leaf in-vector leaf-vector)
                     (bt:release-lock (cl-ds:at leaf-locks leaf))))
                 (iterate
                   (with prediction = (prediction data-point))
                   (for column from 0 below (sl.data:attributes-count target-data))
                   (for difference = (- (sl.data:mref target-data data-point column)
                                        (aref prediction column)))
                   (adjust-predictions data-point column difference))
                 (when parallel
                   (iterate
                     (for leaf in-vector leaf-vector)
                     (bt:release-lock (cl-ds:at leaf-locks leaf)))))
               batch)
      (finally (return ensemble)))))
