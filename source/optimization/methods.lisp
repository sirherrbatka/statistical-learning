(cl:in-package #:statistical-learning.optimization)


(defmethod response ((function squared-error-function)
                     expected
                     function-output)
  (declare (optimize (speed 3) (safety 0)))
  (sl.data:check-data-points expected function-output)
  (iterate
    (declare (type fixnum i))
    (with result = (sl.data:make-data-matrix-like expected))
    (for i from 0 below (sl.data:data-points-count result))
    (setf (sl.data:mref result i 0)
          (- (sl.data:mref expected i 0)
             (sl.data:mref function-output i 0)))
    (finally (return result))))


(defmethod loss ((function squared-error-function)
                 target-data
                 weights
                 &optional split-array)
  (declare (type (or null weights-data-matrix) weights)
           (type sl.data:double-float-data-matrix target-data)
           (type (or null sl.data:split-vector) split-array)
           (optimize (speed 3) (safety 0) (debug 0)))
  (cl-ds.utils:cases ((null weights)
                      (null split-array))
    (let ((left-sum 0.0d0)
          (right-sum 0.0d0)
          (left-count 0)
          (right-count 0))
      (declare (type double-float left-sum right-sum)
               (type sl.data:data-matrix target-data)
               (type fixnum left-count right-count))
      (iterate
        (declare (type fixnum i)
                 (type double-float value))
        (for i from 0 below (sl.data:data-points-count target-data))
        (for value = (sl.data:mref target-data i 0))
        (if (and split-array (eql right (aref split-array i)))
            (setf right-count (1+ right-count)
                  right-sum (+ right-sum value))
            (setf left-count (1+ left-count)
                  left-sum (+ left-sum value))))
      (iterate
        (declare (type double-float
                       left-error right-error
                       left-avg right-avg value)
                 (type fixnum i))
        (with left-error = 0.0d0)
        (with right-error = 0.0d0)
        (with left-avg = (if (zerop left-count)
                             0.0d0
                             (/ left-sum left-count)))
        (with right-avg = (if (zerop right-count)
                              0.0d0
                              (/ right-sum right-count)))
        (for i from 0 below (sl.data:data-points-count target-data))
        (for value = (sl.data:mref target-data i 0))
        (if (and split-array (eql (aref split-array i) right))
            (incf right-error (square (if (null weights)
                                          #1=(- value right-avg)
                                          (* (weight-at weights i) #1#))))
            (incf left-error (square (if (null weights)
                                         #2=(- value left-avg)
                                         (* (weight-at weights i) #2#)))))
        (finally (return (values (if (zerop left-count)
                                     0.0d0
                                     (/ left-error left-count))
                                 (if (zerop right-count)
                                     0.0d0
                                     (/ right-error right-count)))))))))


(defmethod response ((function k-logistic-function)
                     expected
                     sums)
  (declare (optimize (speed 3) (safety 0))
           (type sl.data:double-float-data-matrix sums expected))
  (iterate
    (declare (type fixnum i number-of-classes))
    (with number-of-classes = (sl.data:attributes-count sums))
    (with result = (statistical-learning.data:make-data-matrix-like sums))
    (for i from 0 below (sl.data:data-points-count expected))
    (iterate
      (declare (type fixnum j))
      (for j from 0 below number-of-classes)
      (setf (sl.data:mref result i j)
            (- (if (= (coerce j 'double-float)
                      (sl.data:mref expected i 0))
                   1.0d0
                   0.0d0)
               (sl.data:mref sums i j))))
    (finally (return result))))


(defmethod loss ((function gini-impurity-function)
                 target-data
                 weights
                 &optional split-array)
  (declare (type (or null weights-data-matrix) weights)
           (type sl.data:double-float-data-matrix target-data)
           (type (or null sl.data:split-vector) split-array)
           (optimize (speed 3) (safety 0) (debug 0)))
  (cl-ds.utils:cases ((null split-array)
                      (null weights))
    (iterate
      (declare (type fixnum i)
               (type (simple-array double-float (*))
                     left-sums right-sums)
               (optimize (speed 3) (safety 0) (debug 0)))
      (with number-of-classes = (number-of-classes function))
      (with left-sums = (make-array number-of-classes
                                    :initial-element 0.0d0
                                    :element-type 'double-float))
      (with right-sums = (make-array number-of-classes
                                     :initial-element 0.0d0
                                     :element-type 'double-float))
      (for i from 0 below (sl.data:data-points-count target-data))
      (for target = (the fixnum (truncate (sl.data:mref target-data i 0))))
      (if (and split-array (eql right (aref split-array i)))
          (incf (aref right-sums target) (if (null weights)
                                             1.0d0
                                             (weight-at weights i)))
          (incf (aref left-sums target) (if (null weights)
                                            1.0d0
                                            (weight-at weights i))))
      (finally
       (return (values (vector-impurity left-sums)
                       (vector-impurity right-sums)))))))


(defmethod initialize-instance :after ((function k-logistic-function)
                                       &rest initargs)
  (declare (ignore initargs))
  (let ((number-of-classes (number-of-classes function)))
    (unless (integerp number-of-classes)
      (error 'type-error
             :expected-type 'integer
             :datum number-of-classes))
    (when (< number-of-classes 2)
      (error 'cl-ds:argument-value-out-of-bounds
             :bounds '(>= :number-of-classes 2)
             :value number-of-classes
             :argument :number-of-classes
             :format-control "Classification requires at least 2 classes for classification."))))


(defmethod initialize-instance :after ((function gini-impurity-function)
                                       &rest initargs)
  (declare (ignore initargs))
  (let ((number-of-classes (number-of-classes function)))
    (unless (integerp number-of-classes)
      (error 'type-error
             :expected-type 'integer
             :datum number-of-classes))
    (when (< number-of-classes 2)
      (error 'cl-ds:argument-value-out-of-bounds
             :bounds '(>= :number-of-classes 2)
             :value number-of-classes
             :argument :number-of-classes
             :format-control "Classification requires at least 2 classes for classification."))))
