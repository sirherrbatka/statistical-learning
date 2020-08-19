(cl:in-package #:statistical-learning.optimization)


(declaim (inline weight-at))
(defun weight-at (weights index)
  (declare (type (or null weights-data-matrix) weights)
           (type fixnum index))
  (if (null weights)
      1.0d0
      (sl.data:mref weights index 0)))


(declaim (inline square))
(defun square (x)
  (* x x))


(-> vector-impurity ((simple-array double-float *)) double-float)
(defun vector-impurity (sums)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((length (array-total-size sums))
         (grand-total (iterate
                        (declare (type fixnum i)
                                 (type double-float result))
                        (with result = 0.0d0)
                        (for i from 0 below length)
                        (incf result (row-major-aref sums i))
                        (finally (return result)))))
    (declare (type double-float grand-total))
    (if (zerop grand-total)
        0.0d0
        (iterate
          (declare (type fixnum length i)
                   (type double-float impurity p))
          (with impurity = 0.0d0)
          (for i from 0 below length)
          (for sum = (row-major-aref sums i))
          (when (zerop sum)
            (next-iteration))
          (for p = (/ sum grand-total))
          (incf impurity (* p p))
          (finally (return (- 1.0d0 impurity)))))))
