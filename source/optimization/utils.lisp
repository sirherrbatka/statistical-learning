(cl:in-package #:statistical-learning.optimization)


(declaim (inline weight-at))
(defun weight-at (weights index)
  (declare (type (or null weights-data-matrix) weights)
           (type fixnum index))
  (if (null weights)
      1.0
      (sl.data:mref weights index 0)))


(declaim (inline square))
(defun square (x)
  (* x x))


(-> vector-impurity ((or (simple-array single-float (*)) (simple-array fixnum (*)))) single-float)
(declaim (inline vector-impurity))
(defun vector-impurity (sums)
  (let* ((length (array-total-size sums))
         (grand-total (iterate
                        (declare (type fixnum i)
                                 (type single-float result))
                        (with result = 0.0)
                        (for i from 0 below length)
                        (incf result (row-major-aref sums i))
                        (finally (return result)))))
    (declare (type single-float grand-total))
    (if (zerop grand-total)
        0.0
        (iterate
          (declare (type fixnum length i)
                   (type single-float impurity p))
          (with impurity = 0.0)
          (for i from 0 below length)
          (for sum = (row-major-aref sums i))
          (when (zerop sum)
            (next-iteration))
          (for p = (/ sum grand-total))
          (incf impurity (* p p))
          (finally (return (- 1.0 impurity)))))))



(-> data-point-squared-error ((simple-array single-float (*))
                              sl.data:single-float-data-matrix
                              (or null weights-data-matrix)
                              fixnum)
    single-float)
(defun data-point-squared-error (avg-vector target-data weights i)
  (declare (optimize (speed 3) (safety 0) (debug 0) (space 0)))
  (macrolet ((op (result ii)
               `(let* ((value (sl.data:mref target-data i ,ii))
                       (avg (aref avg-vector ,ii))
                       (error (* weight (square (- value avg)))))
                  (incf ,result error))))
    (iterate
      (declare (type fixnum ii)
               (type single-float result1))
      (with weight = (if (null weights)
                         1.0
                         (weight-at weights i)))
      (with result1 = 0.0)
      (with size = (sl.data:attributes-count target-data))
      (for ii from 0 below size)
      (op result1 ii)
      (finally (return result1 )))))
