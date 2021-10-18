(cl:in-package #:statistical-learning.random)


(declaim (inline random-uniform))
(-> random-uniform (double-float double-float) double-float)
(defun random-uniform (min max)
  (declare (optimize (speed 3) (space 0)
                     (debug 0) (safety 0)
                     (compilation-speed 0))
           (type double-float min max))
  (+ (random (- max min)) min))


(declaim (inline random-gauss))
(-> random-gauss (&optional double-float double-float) double-float)
(defun random-gauss (&optional (mean 0.0d0) (std-dev 0.1d0))
  (declare (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0))
           (type double-float mean std-dev))
  (do* ((rand-u (* 2.0d0 (- 0.5d0 (random 1.0d0)))
                (* 2.0d0 (- 0.5d0 (random 1.0d0))))
        (rand-v (* 2.0d0 (- 0.5d0 (random 1.0d0)))
                (* 2.0d0 (- 0.5d0 (random 1.0d0))))
        (rand-s (+ (* rand-u rand-u)
                   (* rand-v rand-v))
                (+ (* rand-u rand-u)
                   (* rand-v rand-v))))
       ((nor (= 0.0d0 rand-s) (>= rand-s 1.0d0))
        (+ mean
           (* std-dev
              (* rand-u (the double-float (sqrt (/ (* -2.0d0 (log rand-s))
                                                   rand-s)))))))))
