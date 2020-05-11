(cl:in-package #:cl-grf.forest)


(defun total-support (leafs index)
  (iterate
    (for l in-vector leafs)
    (sum (cl-grf.alg:support (aref l index)))))
