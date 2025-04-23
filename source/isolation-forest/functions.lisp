(cl:in-package #:statistical-learning.isolation-forest)


(defun c-factor (n)
  (- (* 2.0 (+ +euler-constant+ (log (- n 1.0))))
     (/ (* 2.0 (- n 1.0))
        n)))
