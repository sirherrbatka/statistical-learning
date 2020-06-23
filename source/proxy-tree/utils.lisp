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


(defun split-treatment (treatment size split-array position)
  (declare (type fixnum size)
           (type sl.data:split-vector treatment split-array))
  (iterate
    (with result = (make-array size
                               :element-type (array-element-type treatment)))
    (with j = 0)
    (for i from 0 below (length split-array))
    (when (eql position (aref split-array i))
      (setf (aref result j) (aref treatment i)
            j (1+ j)))
    (finally (return result))))
