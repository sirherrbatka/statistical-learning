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


(declaim (inline random-uniform))
(-> random-uniform (double-float double-float) double-float)
(defun random-uniform (min max)
  (+ (random (- max min)) min))


(defun gauss-random (&optional (mean 0.0d0) (std-dev 0.1d0))
  "Normal random numbers, with the given mean & standard deviation."
  (do* ((rand-u (* 2 (- 0.5d0 (random 1.0d0)))
                (* 2 (- 0.5d0 (random 1.0d0))))
        (rand-v (* 2 (- 0.5d0 (random 1.0d0)))
                (* 2 (- 0.5d0 (random 1.0d0))))
        (rand-s (+ (* rand-u rand-u) (* rand-v rand-v))
                (+ (* rand-u rand-u) (* rand-v rand-v))))
       ((nor (= 0 rand-s) (>= rand-s 1))
        (+ mean
           (* std-dev
              (* rand-u (sqrt (/ (* -2.0d0 (log rand-s))
                                 rand-s))))))))
