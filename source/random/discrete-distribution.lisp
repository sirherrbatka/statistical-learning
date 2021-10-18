(cl:in-package #:statistical-learning.random)


(defun discrete-distribution (weights)
  (declare (type sl.data:double-float-data-matrix weights))
  (let* ((data-points-count (~>> weights
                                 cl-ds.utils:unfold-table
                                 (count-if-not #'zerop)))
         (probs (make-array data-points-count)))
    (iterate
      (declare (type fixnum i j))
      (with ac = 0.0d0)
      (with j = 0)
      (for i from 0 below (sl.data:data-points-count weights))
      (for weight = (statistical-learning.data:mref weights i 0))
      (when (zerop weight)
        (next-iteration))
      (setf (aref probs j) (cons (incf ac weight)
                                 i))
      (incf j))
    (let ((max (car (last-elt probs))))
      (declare (type double-float max))
      (lambda ()
        (declare (optimize (speed 3) (safety 0)))
        (~>> (cl-ds.utils:lower-bound probs (random max) #'< :key #'car)
             (aref probs)
             cdr)))))
