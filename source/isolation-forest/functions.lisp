(cl:in-package #:statistical-learning.isolation-forest)


(defun c-factor (n)
  (- (* 2.0d0 (+ +euler-constant+ (log (- n 1.0d0))))
     (/ (* 2.0d0 (- n 1.0d0))
        n)))
