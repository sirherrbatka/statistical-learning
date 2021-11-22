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


(-> vector-impurity ((or (simple-array double-float (*)) (simple-array fixnum (*)))) double-float)
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



(-> data-point-squared-error ((simple-array double-float (*))
                              sl.data:double-float-data-matrix
                              (or null weights-data-matrix)
                              fixnum)
    double-float)
(declaim (inline data-point-squared-error))
(defun data-point-squared-error (avg-vector target-data weights i)
  (declare (optimize (speed 3) (safety 0) (debug 0) (space 0)))
  (iterate
    (declare (type fixnum ii)
             (type double-float error avg value result))
    (with weight = (if (null weights)
                       1.0d0
                       (weight-at weights i)))
    (with result = 0.0d0)
    (for ii from 0 below (array-dimension target-data 1))
    (for value = (sl.data:mref target-data i ii))
    (for avg = (aref avg-vector ii))
    (for error = (* weight (square (- value avg))))
    (incf result error)
    (finally (return result))))
