(cl:in-package #:statistical-learning.random)


(defun discrete-distribution (weights)
  (declare (type sl.data:single-float-data-matrix weights))
  (let* ((data-points-count (iterate outer
                              (for i from 0 below (sl.data:data-points-count weights))
                              (iterate
                                (for j from 0 below (sl.data:attributes-count weights))
                                (unless (zerop (sl.data:mref weights i j))
                                  (in outer (sum 1))))))
         (probs (make-array data-points-count)))
    (iterate
      (declare (type fixnum i j))
      (with ac = 0.0)
      (with j = 0)
      (for i from 0 below (sl.data:data-points-count weights))
      (for weight = (statistical-learning.data:mref weights i 0))
      (when (zerop weight)
        (next-iteration))
      (setf (aref probs j) (cons (incf ac weight)
                                 i))
      (incf j))
    (let ((max (car (last-elt probs))))
      (declare (type single-float max))
      (lambda ()
        (declare (optimize (speed 3) (safety 0)))
        (~>> (cl-ds.utils:lower-bound probs (random max) #'< :key #'car)
             (aref probs)
             cdr)))))
