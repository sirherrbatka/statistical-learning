(cl:in-package #:statistical-learning.data)


(declaim (inline attributes-count))
(defun attributes-count (data-matrix)
  (check-data-points data-matrix)
  (array-dimension data-matrix 1))


(declaim (inline data-points-count))
(defun data-points-count (data-matrix)
  (check-data-points data-matrix)
  (array-dimension data-matrix 0))


(defun data-matrix-dimensions (data-matrix)
  (check-data-points data-matrix)
  (array-dimensions data-matrix))


(declaim (inline mref))
(defun mref (data-matrix data-point attribute)
  (aref data-matrix data-point attribute))


(declaim (inline (setf mref)))
(defun (setf mref) (new-value data-matrix data-point attribute)
  (setf (aref data-matrix data-point attribute) new-value))


(-> make-data-matrix (fixnum fixnum &optional t t) data-matrix)
(defun make-data-matrix (data-points-count attributes-count
                         &optional
                           (initial-element 0.0d0)
                           (element-type 'double-float))
  (check-type data-points-count fixnum)
  (check-type attributes-count fixnum)
  (assert (> attributes-count 0))
  (assert (> data-points-count 0))
  (make-array `(,data-points-count ,attributes-count)
              :initial-element initial-element
              :element-type element-type))


(-> sample (data-matrix &key
                        (:data-points (or null (simple-array fixnum (*))))
                        (:attributes (or null (simple-array fixnum (*)))))
    data-matrix)
(defun sample (data-matrix &key data-points attributes)
  (declare (optimize (speed 3) (safety 0)))
  (check-data-points data-matrix)
  (dispatch-data-matrix (data-matrix)
    (cl-ds.utils:cases ((null attributes)
                        (null data-points))
      (when (and (null attributes)
                 (null data-points))
        (return-from sample data-matrix))
      (iterate
        (declare (type fixnum i attributes-count data-points-count))
        (with attributes-count = (if (null attributes)
                                     (attributes-count data-matrix)
                                     (length attributes)))
        (with data-points-count = (if (null data-points)
                                      (data-points-count data-matrix)
                                      (length data-points)))
        (with result = (make-data-matrix data-points-count
                                         attributes-count
                                         0.0d0
                                         (array-element-type data-matrix)))
        (for i from 0 below data-points-count)
        (iterate
          (declare (type fixnum j))
          (for j from 0 below attributes-count)
          (setf (mref result i j)
                (mref data-matrix
                      (if (null data-points) i (aref data-points i))
                      (if (null attributes) j (aref attributes j)))))
        (finally (return result))))))


(declaim (inline map-data-matrix))
(defun map-data-matrix (function data-matrix &optional in-place)
  (declare (optimize (speed 3) (safety 0)))
  (check-type data-matrix double-float-data-matrix)
  (dispatch-data-matrix (data-matrix)
    (lret ((result (if in-place
                       data-matrix
                       (make-data-matrix
                        (data-points-count data-matrix)
                        (attributes-count data-matrix)))))
      (iterate
        (declare (type fixnum i))
        (for i from 0 below (array-total-size data-matrix))
        (setf (row-major-aref result i)
              (funcall function (row-major-aref data-matrix i)))))))


(defun make-data-matrix-like (data-matrix &optional (initial-element 0.0d0))
  (check-type data-matrix data-matrix)
  (make-array (array-dimensions data-matrix)
              :element-type (array-element-type data-matrix)
              :initial-element initial-element))


(declaim (inline reduce-data-points))
(-> reduce-data-points (t double-float-data-matrix
                          &key
                          (:attributes (or null (simple-array fixnum (*))))
                          (:data-points (or null (simple-array fixnum (*)))))
    double-float-data-matrix)
(defun reduce-data-points (function data &key attributes data-points)
  (declare (optimize (speed 3) (safety 0)))
  (iterate
    (declare (type fixnum i first-point attributes-count data-points-count)
             (type double-float-data-matrix result))
    (with data-points-count = (if (null data-points)
                                  (data-points-count data)
                                  (length data-points)))
    (with attributes-count = (if (null attributes)
                                 (attributes-count data)
                                 (length attributes)))
    (with result = (make-data-matrix 1 attributes-count))
    (with first-point = (if (null data-points)
                            0
                            (length data-points)))
    (for i from 0 below attributes-count)
    (for attribute = (if (null attributes)
                         i
                         (aref attributes i)))
    (setf (sl.data:mref result 0 i)
          (sl.data:mref data first-point attribute))
    (iterate
      (declare (type fixnum j k))
      (for j from 1 below data-points-count)
      (for k = (if (null data-points)
                   j
                   (aref data-points j)))
      (setf (sl.data:mref result 0 i)
            (funcall function
                     (sl.data:mref result 0 i)
                     (sl.data:mref data k attribute))))
    (finally (return result))))


(-> maxs (double-float-data-matrix
           &key
           (:attributes (or null (simple-array fixnum (*))))
           (:data-points (or null (simple-array fixnum (*)))))
    double-float-data-matrix)
(declaim (inline maxs))
(defun maxs (data &key data-points attributes)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((attributes-count (if (null attributes)
                               (attributes-count data)
                               (length attributes)))
         (data-points-count (if (null data-points)
                                (data-points-count data)
                                (length data-points)))
         (result-max (make-data-matrix 1 attributes-count))
         (first-point (if (null data-points)
                          0
                          (aref data-points 0))))
    (declare (type fixnum attributes-count data-points-count first-point)
             (type sl.data:double-float-data-matrix result-max))
    (iterate
      (declare (type fixnum i attribute))
      (for i from 0 below attributes-count)
      (for attribute = (if (null attributes)
                           i
                           (aref attributes i)))
      (setf (sl.data:mref result-max 0 i) (sl.data:mref data first-point attribute)))
    (iterate
      (declare (type fixnum j k1))
      (for j from 1 below data-points-count)
      (for k1 = (if (null data-points) j (aref data-points j)))
      (iterate
        (declare (type fixnum i1))
        (for i1 from 0 below attributes-count by 1)
        (let ((attribute (if (null attributes)
                             i1
                             (aref attributes i1))))
          (maxf (sl.data:mref result-max 0 i1) (sl.data:mref data k1 attribute))))
      (finally (return result-max)))))


(-> mins/maxs (double-float-data-matrix
               &key
               (:attributes (or null (simple-array fixnum (*))))
               (:data-points (or null (simple-array fixnum (*)))))
    cons)
(declaim (inline mins/maxs))
(defun mins/maxs (data &key data-points attributes)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((attributes-count (if (null attributes)
                              (attributes-count data)
                              (length attributes)))
         (data-points-count (if (null data-points)
                                (data-points-count data)
                                (length data-points)))
         (result-min (make-data-matrix 1 attributes-count))
         (result-max (make-data-matrix 1 attributes-count))
         (first-point (if (null data-points)
                            0
                            (aref data-points 0))))
    (declare (type fixnum attributes-count data-points-count first-point)
             (type sl.data:double-float-data-matrix result-min result-max))
    (iterate
      (declare (type fixnum i attribute))
      (for i from 0 below attributes-count)
      (for attribute = (if (null attributes)
                           i
                           (aref attributes i)))
      (setf (sl.data:mref result-max 0 i) (sl.data:mref data first-point attribute)
            (sl.data:mref result-min 0 i) (sl.data:mref data first-point attribute)))
    (iterate
      (declare (type fixnum j k1))
      (for j from 1 below data-points-count)
      (for k1 = (if (null data-points) j (aref data-points j)))
      (iterate
        (declare (type fixnum i1))
        (for i1 from 0 below attributes-count by 1)
        (let* ((attribute (if (null attributes)
                               i1
                               (aref attributes i1)))
               (value (sl.data:mref data k1 attribute)))
          (minf (sl.data:mref result-min 0 i1) value)
          (maxf (sl.data:mref result-max 0 i1) value)))
      (finally (return (cons result-min result-max))))))


(-> mins (double-float-data-matrix
           &key
           (:attributes (or null (simple-array fixnum (*))))
           (:data-points (or null (simple-array fixnum (*)))))
    double-float-data-matrix)
(declaim (inline mins))
(defun mins (data &key data-points attributes)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((attributes-count (if (null attributes)
                               (attributes-count data)
                               (length attributes)))
         (data-points-count (if (null data-points)
                                (data-points-count data)
                                (length data-points)))
         (result-min (make-data-matrix 1 attributes-count))
         (first-point (if (null data-points)
                          0
                          (aref data-points 0))))
    (declare (type fixnum attributes-count data-points-count first-point)
             (type sl.data:double-float-data-matrix result-min))
    (iterate
      (declare (type fixnum i attribute))
      (for i from 0 below attributes-count)
      (for attribute = (if (null attributes)
                           i
                           (aref attributes i)))
      (setf (sl.data:mref result-min 0 i) (sl.data:mref data first-point attribute)))
    (iterate
      (declare (type fixnum j k1))
      (for j from 1 below data-points-count)
      (for k1 = (if (null data-points) j (aref data-points j)))
      (iterate
        (declare (type fixnum i1))
        (for i1 from 0 below attributes-count by 1)
        (let ((attribute (if (null attributes)
                             i1
                             (aref attributes i1))))
          (minf (sl.data:mref result-min 0 i1) (sl.data:mref data k1 attribute))))
      (finally (return result-min)))))


(-> split (data-matrix fixnum split-vector t (or null fixnum)) data-matrix)
(defun split (data-matrix length split-array position skipped-column)
  (declare (optimize (speed 3) (debug 0) (safety 0)))
  (dispatch-data-matrix (data-matrix)
    (cl-ds.utils:cases ((null skipped-column))
      (bind-data-matrix-dimensions
          ((data-points-count attributes-count data-matrix))
        (lret ((result (make-array `(,length ,(if (null skipped-column)
                                                  attributes-count
                                                  (1- attributes-count)))
                                   :element-type (array-element-type data-matrix))))
          (iterate
            (declare (type fixnum j i))
            (with j = 0)
            (for i from 0 below data-points-count)
            (when (eql position (aref split-array i))
              (iterate
                (declare (type fixnum k p))
                (with p = 0)
                (for k from 0 below attributes-count)
                (when (eql skipped-column k)
                  (next-iteration))
                (setf (mref result j p)
                      (mref data-matrix i k)
                      p (1+ p))
                (finally (assert (= p (array-dimension result 1)))))
              (incf j))
            (finally (assert (= j length)))))))))


(-> data-min/max (sl.data:double-float-data-matrix
                  fixnum
                  (simple-array fixnum (*)))
    (values double-float double-float))
(defun data-min/max (data attribute data-points)
  (declare (type statistical-learning.data:data-matrix data)
           (type fixnum attribute)
           (optimize (speed 3) (safety 0)))
  (iterate
    (declare (type double-float min max element)
             (type fixnum i))
    (with min = (sl.data:mref data (aref data-points 0) attribute))
    (with max = min)
    (with length = (length data-points))
    (for i from 1 below length)
    (for element = (sl.data:mref data (aref data-points i) attribute))
    (cond ((< max element) (setf max element))
          ((> min element) (setf min element)))
    (finally (return (values min max)))))
