(cl:in-package #:statistical-learning.isolation-forest)


(-> generate-point (sl.data:double-float-data-matrix
                    (simple-array fixnum (*))
                    (simple-array fixnum (*)))
    split-point)
(defun generate-point (data samples attributes)
  (iterate
    (with sample = (sl.data:sample data
                                   :data-points samples
                                   :attributes attributes))
    (with attributes-count = (length attributes))
    (with min = (calculate-mins sample))
    (with max = (calculate-maxs sample))
    (with normals = (sl.data:make-data-matrix 1 attributes-count))
    (for i from 0 below attributes-count)
    (setf (sl.data:mref normals 0 i) (sl.common:gauss-random))
    (sum (* (sl.data:mref normals 0 i)
            (random-in-range (sl.data:mref min 0 i)
                             (sl.data:mref max 0 i)))
         into dot-product)
    (finally (return (make-split-point
                      :normals normals
                      :dot-product dot-product)))))


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
  (declare (optimize (speed 0) (safety 3) (debug 3)))
  (iterate
    (for i from 0 below (length attributes))
    (for attribute = (aref attributes i))
    (sum (* (sl.data:mref first first-point attribute)
            (sl.data:mref second second-point i)))))


(defun rightp (split-point attributes data-point data)
  (declare (type sl.data:double-float-data-matrix data)
           (type (simple-array fixnum (*)) attributes)
           (type fixnum data-point))
  (let ((dot-product (split-point-dot-product split-point))
        (normals (split-point-normals split-point)))
    (< (wdot data normals data-point 0 attributes)
       dot-product)))


(defun calculate-mins (data-matrix)
  (sl.data:reduce-data-points #'min data-matrix))


(defun calculate-maxs (data-matrix)
  (sl.data:reduce-data-points #'max data-matrix))


(defun global-min/max (mins maxs)
  (iterate
    (for i from 0 below (sl.data:attributes-count mins))
    (for min = (sl.data:mref mins 0 i))
    (for max = (sl.data:mref maxs 0 i))
    (finding (list min max) maximizing (- max min))))


(defun make-normals (count)
  (iterate
    (with result = (sl.data:make-data-matrix 1 count))
    (for i from 0 below count)
    (setf (sl.data:mref result 0 i) (sl.common:gauss-random))
    (finally (return result))))
