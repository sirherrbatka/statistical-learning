(cl:in-package #:statistical-learning.optimization)


(defmethod response ((function squared-error-function)
                     expected
                     function-output)
  (declare (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0))
           (type sl.data:double-float-data-matrix
                 expected
                 function-output))
  (sl.data:check-data-points expected function-output)
  (iterate
    (declare (type fixnum i)
             (type sl.data:double-float-data-matrix result))
    (with result = (sl.data:make-data-matrix-like expected))
    (for i from 0 below (sl.data:data-points-count result))
    (iterate
      (declare (type fixnum ii))
      (for ii from 0 below (sl.data:attributes-count function-output))
      (setf (sl.data:mref result i ii)
            (- (the double-float (sl.data:mref expected i ii))
               (the double-float (sl.data:mref function-output i ii)))))
    (finally (return result))))


(defmethod loss ((function squared-error-function)
                 target-data
                 weights
                 &optional split-array)
  (declare (type (or null weights-data-matrix) weights)
           (type sl.data:double-float-data-matrix target-data)
           (type (or null sl.data:split-vector) split-array)
           (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0)))
  (let* ((target-data-width (sl.data:attributes-count target-data))
         (left-sum (make-array target-data-width :element-type 'double-float
                                                 :initial-element 0.0d0))
         (right-sum (make-array target-data-width :element-type 'double-float
                                                  :initial-element 0.0d0))
         (length (sl.data:data-points-count target-data))
         (left-count 0)
         (right-count 0))
    (declare (type (simple-array double-float (*)) left-sum right-sum)
             (type fixnum left-count right-count))
    (sl.data:dispatch-data-matrix (target-data)
      (cl-ds.utils:cases ((null split-array)
                          (null weights))
        (iterate
          (declare (type fixnum i))
          (for i from 0 below length)
          (for rightp = (and split-array (eql right (aref split-array i))))
          (if rightp
              (progn (incf right-count)
                     (iterate
                       (declare (type fixnum ii)
                                (type double-float value))
                       (for ii from 0 below target-data-width)
                       (for value = (sl.data:mref target-data i ii))
                       (incf (aref right-sum ii) value)))
              (progn (incf left-count)
                     (iterate
                       (declare (type fixnum ii)
                                (type double-float value))
                       (for ii from 0 below target-data-width)
                       (for value = (sl.data:mref target-data i ii))
                       (incf (aref left-sum ii) value)))))
        (iterate
          (declare (type double-float
                         left-error right-error)
                   (type (simple-array double-float (*))
                         left-avg right-avg)
                   (type fixnum i))
          (with left-error = 0.0d0)
          (with right-error = 0.0d0)
          (with right-avg = (sl.data:vector-avg right-sum right-count))
          (with left-avg = (sl.data:vector-avg left-sum left-count))
          (for i from 0 below length)
          (for rightp = (and split-array (eql right (aref split-array i))))
          (if rightp
              (incf right-error (data-point-squared-error right-avg
                                                          target-data
                                                          weights
                                                          i))
              (incf left-error (data-point-squared-error left-avg
                                                         target-data
                                                         weights
                                                         i)))
          (finally (return (values (if (zerop left-count)
                                       0.0d0
                                       (/ left-error left-count))
                                   (if (zerop right-count)
                                       0.0d0
                                       (/ right-error right-count))))))))))


(defmethod response ((function k-logistic-function)
                     expected
                     sums)
  (declare (optimize (speed 3) (safety 0)
                     (space 0) (debug 0)
                     (compilation-speed 0))
           (type sl.data:double-float-data-matrix sums expected))
  (iterate
    (declare (type fixnum i number-of-classes)
             (type sl.data:double-float-data-matrix result))
    (with number-of-classes = (sl.data:attributes-count sums))
    (with result = (sl.data:make-data-matrix-like sums))
    (for i from 0 below (sl.data:data-points-count expected))
    (iterate
      (declare (type fixnum j))
      (for j from 0 below number-of-classes)
      (setf (sl.data:mref result i j)
            (- (if (= j (sl.data:mref expected i 0))
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
           (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0)))
  (cl-ds.utils:cases ((null split-array))
    (sl.data:dispatch-data-matrix (target-data)
      (if (null weights)
          (iterate
            (declare (type fixnum i target1 length)
                     (type (simple-array fixnum (*))
                           left-sums right-sums))
            (with number-of-classes = (number-of-classes function))
            (with left-sums = (make-array number-of-classes
                                          :initial-element 0
                                          :element-type 'fixnum))
            (with right-sums = (make-array number-of-classes
                                           :initial-element 0
                                           :element-type 'fixnum))
            (with length = (sl.data:data-points-count target-data))
            (for i from 0 below length)
            (for target1 = (truncate (statistical-learning.data:mref target-data i 0)))
            (if (and split-array (eql right (aref split-array i)))
                (incf (aref right-sums target1))
                (incf (aref left-sums target1)))
            (finally
             (return (values (vector-impurity left-sums)
                             (vector-impurity right-sums)))))
          (iterate
            (declare (type fixnum i)
                     (type (simple-array double-float (*))
                           left-sums right-sums))
            (with number-of-classes = (number-of-classes function))
            (with left-sums = (make-array number-of-classes
                                          :initial-element 0.0d0
                                          :element-type 'double-float))
            (with right-sums = (make-array number-of-classes
                                           :initial-element 0.0d0
                                           :element-type 'double-float))
            (for i from 0 below (sl.data:data-points-count target-data))
            (for target = (truncate (statistical-learning.data:mref target-data i 0)))
            (if (and split-array (eql right (aref split-array i)))
                (incf (aref right-sums target) (weight-at weights i))
                (incf (aref left-sums target) (weight-at weights i)))
            (finally
             (return (values (vector-impurity left-sums)
                             (vector-impurity right-sums)))))))))


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
