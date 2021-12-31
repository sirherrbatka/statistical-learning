(cl:in-package #:statistical-learning.gradient-descent-refine)


(defmethod sl.ensemble:refine-trees ((algorithm parameters)
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
         ((:flet prediction (data-point))
          (iterate
            (with leaf-vector = (aref leafs data-point 0))
            (with number-of-trees = (length trees))
            (with sums = (make-array number-of-trees
                                     :element-type 'double-float
                                     :initial-element 0.0d0))
            (for i from 0 below number-of-trees)
            (for tree = (aref trees i))
            (for weight = (sl.tp:weight tree))
            (for leaf = (aref leaf-vector i))
            (for predictions = (sl.tp:predictions leaf))
            (iterate
              (for ii from 0 below (sl.data:attributes-count predictions))
              (for pred = (sl.data:mref predictions 0 ii))
              (incf (aref sums ii) (* weight pred)))
            (finally (return (cl-ds.utils:transform (rcurry #'/ total-weight)
                                                    sums))))))
    (iterate
      (for epoch from epochs downto 1)
      (setf shrinkage (* initial-shrinkage (/ epoch epochs)))
      (sl.data:reshuffle indexes)
      (iterate
        (for i from 0 below (if sample-size
                                (min sample-size
                                     (length indexes))
                                (length indexes)))
        (for data-point = (aref indexes i))
        (iterate
          (with prediction = (prediction data-point))
          (for column from 0 below (sl.data:attributes-count target-data))
          (for difference = (- (sl.data:mref target-data data-point column)
                               (aref prediction column)))
          (adjust-predictions data-point column difference)))
      (finally (return ensemble)))))
