(cl:in-package #:statistical-learning.gradient-descent-refine)


(defmethod sl.ensemble:refine-trees ((algorithm parameters)
                                     ensemble
                                     train-data
                                     target-data)
  (refine-trees (~> ensemble sl.mp:parameters sl.ensemble:tree-parameters)
                algorithm ensemble
                train-data target-data))


(defmethod refine-trees ((parameters sl.dt:classification)
                         algorithm
                         ensemble
                         train-data
                         target-data)
  (bind ((optimized-function (sl.opt:optimized-function parameters))
         (number-of-classes (sl.opt:number-of-classes optimized-function))
         (data-points-count (sl.data:data-points-count target-data))
         (result (sl.data:make-data-matrix data-points-count
                                           number-of-classes)))
    (check-type target-data sl.data:double-float-data-matrix)
    (check-type train-data sl.data:data-matrix)
    (check-type result sl.data:double-float-data-matrix)
    (iterate
      (declare (optimize (speed 3) (safety 0))
               (type fixnum i))
      (for i from 0 below data-points-count)
      (setf (sl.data:mref result i
                          (the fixnum (truncate (sl.data:mref target-data i 0))))
            1.0d0))
    (refine-implementation algorithm ensemble
                           train-data result)))


(defmethod refine-trees ((tree-parameters sl.dt:regression)
                         algorithm
                         ensemble
                         train-data
                         target-data)
  (refine-implementation algorithm ensemble
                         train-data target-data))
