(cl:in-package #:sl.proxy-tree)


(defun train/adjust (attributes training-state)
  (bind ((data-points-count (~> training-state
                                sl.mp:train-data
                                sl.data:data-points-count))
         (half-data-points (truncate data-points-count 2))
         (randomized-indexes (~> data-points-count
                                 sl.data:iota-vector
                                 sl.data:reshuffle))
         (division (sl.mp:sample-training-state
                    training-state
                    :train-attributes attributes
                    :data-points (take half-data-points randomized-indexes)))
         (values (sl.mp:sample-training-state
                  training-state
                  :data-points (drop half-data-points randomized-indexes))))
    (values division values)))
