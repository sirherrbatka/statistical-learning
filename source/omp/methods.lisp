(cl:in-package #:statistical-learning.omp)


(defmethod prune-trees ((parameters sl.perf:classification)
                        omp
                        ensemble
                        trees
                        train-data
                        target-data)
  (bind ((optimized-function (sl.opt:optimized-function parameters))
         (number-of-classes (sl.opt:number-of-classes optimized-function))
         (data-points-count (sl.data:data-points-count target-data)))
    (check-type train-data sl.data:double-float-data-matrix)
    (if (= number-of-classes 2)
        (prune-trees-implementation omp ensemble trees train-data target-data)
        (let ((result (sl.data:make-data-matrix data-points-count
                                                number-of-classes)))
          (check-type result sl.data:double-float-data-matrix)
          (iterate
            (declare (optimize (speed 3) (safety 0))
                     (type fixnum i))
            (for i from 0 below data-points-count)
            (setf (sl.data:mref result i
                                (the fixnum (truncate (sl.data:mref target-data i 0))))
                  1.0d0))
          (prune-trees-implementation omp ensemble trees train-data result)))))


(defmethod prune-trees ((parameters sl.perf:regression)
                        omp
                        ensemble
                        trees
                        train-data
                        target-data)
  (prune-trees-implementation omp ensemble trees train-data target-data))


(defmethod sl.ensemble:prune-trees ((algorithm orthogonal-matching-pursuit)
                                    (ensemble-model sl.ensemble:ensemble-model)
                                    train-data
                                    target-data)
  (bind ((ensemble (sl.mp:parameters ensemble-model))
         (parameters (sl.ensemble:tree-parameters ensemble)))
    (cl-ds.utils:quasi-clone ensemble-model
                             :trees (prune-trees parameters
                                                 algorithm
                                                 ensemble
                                                 (sl.ensemble:trees ensemble-model)
                                                 train-data
                                                 target-data))))
