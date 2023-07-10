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
(declaim (inline vector-impurity))
(defun vector-impurity (sums)
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
(declaim (notinline data-point-squared-error))
(defun data-point-squared-error (avg-vector target-data weights i)
  (declare (optimize (speed 0) (safety 3) (debug 3) (space 0)))
  (macrolet ((op (result ii)
               `(let* ((value (sl.data:mref target-data i ,ii))
                       (avg (aref avg-vector ,ii))
                       (error (* weight (square (- value avg)))))
                  (incf ,result error))))
    (iterate
      (declare (type fixnum ii)
               (type double-float result1))
      (with weight = (if (null weights)
                         1.0d0
                         (weight-at weights i)))
      (with result1 = 0.0d0)
      (with size = (sl.data:attributes-count target-data))
      (for ii from 0 below size)
      (op result1 ii)
      (finally (return result1 )))))
