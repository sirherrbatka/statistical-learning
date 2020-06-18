(cl:in-package #:statistical-learning.random)


(defun discrete-distribution (weights)
  (statistical-learning.data:check-data-points weights)
  (bind ((data-points (sl.data:data-points-count weights))
         (probs (make-array data-points :element-type 'double-float
                            :initial-element 0.0d0)))
    (iterate
      (with ac = 0.0d0)
      (for i from 0 below data-points)
      (setf (aref probs i) (incf ac (statistical-learning.data:mref weights i 0))))
    (let ((max (last-elt probs)))
      (declare (type double-float max))
      (lambda ()
        (declare (optimize (speed 3) (safety 0)))
        (cl-ds.utils:lower-bound probs (random max) #'<)))))
