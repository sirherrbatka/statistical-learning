(cl:in-package #:statistical-learning.random)


(declaim (inline random-uniform))
(-> random-uniform (single-float single-float) single-float)
(defun random-uniform (min max)
  (declare (optimize (speed 3) (space 0)
                     (debug 0) (safety 0)
                     (compilation-speed 0))
           (type single-float min max))
  (+ (random (- max min)) min))


(declaim (inline random-gauss))
(-> random-gauss (&optional single-float single-float) single-float)
(defun random-gauss (&optional (mean 0.0) (std-dev 0.1))
  (declare (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0))
           (type single-float mean std-dev))
  (do* ((rand-u (* 2.0 (- 0.5 (random 1.0)))
                (* 2.0 (- 0.5 (random 1.0))))
        (rand-v (* 2.0 (- 0.5 (random 1.0)))
                (* 2.0 (- 0.5 (random 1.0))))
        (rand-s (+ (* rand-u rand-u)
                   (* rand-v rand-v))
                (+ (* rand-u rand-u)
                   (* rand-v rand-v))))
       ((nor (= 0.0 rand-s) (>= rand-s 1.0))
        (+ mean
           (* std-dev
              (* rand-u (the single-float (sqrt (/ (* -2.0 (log rand-s))
                                                   rand-s)))))))))
