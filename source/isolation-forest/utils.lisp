(cl:in-package #:statistical-learning.isolation-forest)


(-> scale-double-float
    (double-float double-float double-float double-float double-float)
    double-float)
(defun scale-double-float (value range scale-min min max)
  (if (= max min)
      value
      (+ (* range (/ (- value min) (- max min)))
         scale-min)))


(-> wdot (sl.data:double-float-data-matrix
          sl.data:double-float-data-matrix
          fixnum
          fixnum
          (simple-array fixnum (*)))
    double-float)
(defun wdot (first second first-point second-point attributes)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (iterate
    (declare (type fixnum i)
             (type double-float result))
    (with result = 0.0d0)
    (for i from 0 below (length attributes))
    (for attribute = (aref attributes i))
    (incf result
          (* (sl.data:mref first first-point attribute)
             (sl.data:mref second second-point i)))
    (finally (return result))))


(-> rightp (split-point
            (simple-array fixnum (*))
            fixnum
            sl.data:double-float-data-matrix)
    boolean)
(defun rightp (split-point attributes data-point data)
  (declare (optimize (speed 3) (safety 0)))
  (let ((dot-product (split-point-dot-product split-point))
        (normals (split-point-normals split-point)))
    (< (wdot data normals data-point 0 attributes)
       dot-product)))


(defun calculate-mins (data-matrix samples attributes)
  (sl.data:reduce-data-points #'min
                              data-matrix
                              :data-points samples
                              :attributes attributes))


(defun calculate-maxs (data-matrix samples attributes)
  (sl.data:reduce-data-points #'max
                              data-matrix
                              :data-points samples
                              :attributes attributes))


(defun make-normals (count)
  (iterate
    (with result = (sl.data:make-data-matrix 1 count))
    (for i from 0 below count)
    (setf (sl.data:mref result 0 i) (sl.common:gauss-random
                                     0.0d0 1.0d0))
    (finally (return result))))
