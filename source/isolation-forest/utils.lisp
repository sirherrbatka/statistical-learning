(cl:in-package #:statistical-learning.isolation-forest)


(-> generate-point (sl.data:double-float-data-matrix
                    (simple-array fixnum (*))
                    (simple-array fixnum (*))
                    double-float
                    sl.data:double-float-data-matrix
                    sl.data:double-float-data-matrix
                    t)
    sl.data:double-float-data-matrix)
(defun generate-point (data samples attributes
                       depth-ratio min max gaussian-state)
  (iterate
    (with random-sample = (make-array
                           1
                           :element-type 'fixnum
                           :initial-element (aref samples (random (length samples)))))
    (with p = (sl.data:sample data
                              :data-points random-sample
                              :attributes attributes))
    (for i from 0 below (sl.data:attributes-count p))
    (for attribute = (aref attributes i))
    (incf (sl.data:mref p 0 i)
          (* (sl.common:gauss-random gaussian-state)
             depth-ratio
             (if (= (sl.data:mref max 0 attribute)
                    (sl.data:mref min 0 attribute))
                 0.5d0
                 (/ (- (sl.data:mref max 0 attribute)
                       (sl.data:mref min 0 attribute))
                    2.0d0))))
    (finally (return p))))


(-> scale-double-float
    (double-float double-float double-float double-float double-float)
    double-float)
(defun scale-double-float (value range scale-min min max)
  (if (= max min)
      value
      (+ (* range
            (/ (- value min)
               (- max min)))
         scale-min)))


(-> wdot (sl.data:double-float-data-matrix
          sl.data:double-float-data-matrix
          fixnum
          fixnum
          (simple-array fixnum (*))
          sl.data:double-float-data-matrix
          sl.data:double-float-data-matrix
          double-float
          double-float)
    double-float)
(defun wdot (first second first-point second-point attributes
             mins maxs global-min global-max)
  (declare (optimize (speed 0) (safety 3) (debug 3)))
  (iterate
    (with range = (- global-max global-min))
    (for i from 0 below (length attributes))
    (for attribute = (aref attributes i))
    (sum (* (scale-double-float (sl.data:mref first
                                              first-point
                                              attribute)
                                range
                                global-min
                                (sl.data:mref mins 0 attribute)
                                (sl.data:mref maxs 0 attribute))
            (sl.data:mref second second-point attribute)))))


(defun rightp (split-point normals data-point data
               mins maxs global-min global-max)
  (declare (type isolation-forest-split-point split-point)
           (type sl.data:double-float-data-matrix normals data)
           (type fixnum data-point))
  (bind ((dot-product (isolation-forest-split-point-dot-product
                       split-point))
         (attributes (isolation-forest-split-point-attributes
                      split-point)))
    (> (wdot normals data 0 data-point attributes
             mins maxs global-min global-max)
       dot-product)))


(defun calculate-mins (data-matrix)
  (sl.data:reduce-data-points #'min data-matrix))


(defun calculate-maxs (data-matrix)
  (sl.data:reduce-data-points #'max data-matrix))
