(cl:in-package #:statistical-learning.common)


(declaim (inline random-uniform))
(-> random-uniform (double-float double-float) double-float)
(defun random-uniform (min max)
  (+ (random (- max min)) min))


(-> side-of-line (double-float double-float
                  double-float double-float
                  double-float double-float)
    double-float)
(declaim (inline side-of-line))
(defun side-of-line (p1x p1y p2x p2y p3x p3y)
  (declare (type double-float p1x p1y p2x p2y p3x p3y)
           (optimize (speed 3) (safety 0)))
  (- (* (- p1x p3x) (- p2y p3y))
     (* (- p2x p3x) (- p1y p3y))))


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
