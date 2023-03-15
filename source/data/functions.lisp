(cl:in-package #:statistical-learning.data)


(declaim (inline data))
(defun data (data-matrix)
  (declare (type data-matrix data-matrix))
  (if (typep data-matrix 'universal-data-matrix)
      (universal-data-matrix-data data-matrix)
      (double-float-data-matrix-data data-matrix)))


(declaim (inline index))
(defun index (data-matrix)
  (declare (type data-matrix data-matrix))
  (if (typep data-matrix 'universal-data-matrix)
      (universal-data-matrix-index data-matrix)
      (double-float-data-matrix-index data-matrix)))


(declaim (inline data-matrix-element-type))
(defun data-matrix-element-type (data-matrix)
  (~> data-matrix data array-element-type))


(declaim (inline attributes-count))
(defun attributes-count (data-matrix)
  (check-data-points data-matrix)
  (array-dimension (data data-matrix) 1))


(declaim (inline data-points-count))
(defun data-points-count (data-matrix)
  (check-data-points data-matrix)
  (array-dimension (index data-matrix) 0))


(defun data-matrix-dimensions (data-matrix)
  (check-data-points data-matrix)
  (list (array-dimension (index data-matrix) 0)
        (array-dimension (data data-matrix) 1)))


(declaim (inline mref))
(defun mref (data-matrix data-point attribute)
  (aref (data data-matrix)
        (aref (index data-matrix) data-point)
        attribute))


(declaim (inline (setf mref)))
(defun (setf mref) (new-value data-matrix data-point attribute)
  (setf (aref (data data-matrix)
              (aref (index data-matrix) data-point)
              attribute)
        new-value))


(defun make-iota-vector (length)
  (iterate
    (with result = (make-array length :element-type 'fixnum))
    (for i from 0 below length)
    (setf (aref result i) i)
    (finally (return result))))


(declaim (inline data-matrix-constructor))
(defun data-matrix-constructor (data-matrix)
  (check-type data-matrix data-matrix)
  (let ((element-type (data-matrix-element-type data-matrix)))
    (econd ((eq element-type 'double-float) #'make-double-float-data-matrix)
           ((eq element-type t) #'make-universal-data-matrix))))


(-> make-data-matrix (fixnum fixnum &optional t t) data-matrix)
(defun make-data-matrix (data-points-count attributes-count
                         &optional
                           (initial-element 0.0d0)
                           (element-type 'double-float))
  (check-type data-points-count fixnum)
  (check-type attributes-count fixnum)
  (assert (> attributes-count 0))
  (assert (> data-points-count 0))
  (econd ((eq element-type 'double-float)
          (make-double-float-data-matrix
           :data (make-array `(,data-points-count ,attributes-count)
                             :initial-element initial-element
                             :element-type 'double-float)
           :index (make-iota-vector data-points-count)))
         ((eq element-type t)
          (make-universal-data-matrix
           :data (make-array `(,data-points-count ,attributes-count)
                             :initial-element initial-element
                             :element-type t)
           :index (make-iota-vector data-points-count)))))


(defun wrap (input)
  (if (typep input 'data-matrix)
      input
      (progn
        (check-type input (or (simple-array double-float (* *))
                              (simple-array t (* *))))
        (let ((element-type (array-element-type input))
              (data-points-count (array-dimension input 0)))
          (econd ((eq element-type 'double-float)
                  (make-double-float-data-matrix
                   :data input
                   :index (make-iota-vector data-points-count)))
                 ((eq element-type t)
                  (make-universal-data-matrix
                   :data input
                   :index (make-iota-vector data-points-count))))))))


(-> sample (data-matrix &key
                        (:data-points (or null vector))
                        (:attributes (or null vector)))
    data-matrix)
(defun sample (data-matrix &key data-points attributes)
  (declare (optimize (speed 3) (debug 0) (safety 0)))
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
                                         (data-matrix-element-type data-matrix)))
        (for i from 0 below data-points-count)
        (iterate
          (declare (type fixnum j))
          (for j from 0 below attributes-count)
          (setf (mref result i j)
                (mref data-matrix
                      (if (null data-points) i (aref data-points i))
                      (if (null attributes) j (aref attributes j)))))
        (finally (return result))))))


(defun draw-random-data-points-subset (sample-size data-matrix &rest rest)
  (if (null sample-size)
      (cons data-matrix rest)
      (bind ((data-points-count (data-points-count data-matrix))
             (effective-sample-size (min sample-size data-points-count))
             (selected (map-into (make-array effective-sample-size :element-type 'fixnum)
                                 (cl-ds.utils:lazy-shuffle 0 data-points-count)))
             ((:flet impl (data-matrix))
              (sample data-matrix :data-points selected)))
        (mapcar #'impl (cons data-matrix rest)))))


(defun make-data-matrix-like (data-matrix &optional (initial-element 0.0d0))
  (check-type data-matrix data-matrix)
  (make-data-matrix (data-points-count data-matrix)
                    (attributes-count data-matrix)
                    initial-element
                    (data-matrix-element-type data-matrix)))


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
           (:data-points vector))
    double-float-data-matrix)
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
               (:data-points vector))
    cons)
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
           (:data-points vector))
    double-float-data-matrix)
(defun mins (data &key data-points attributes)
  (declare (optimize (speed 3) (debug 0) (safety 0)))
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


(-> split (list fixnum split-vector t) list)
(defun split (data-matrixes length split-array position)
  (if (endp data-matrixes)
      nil
      (let ((old-index (~> data-matrixes first index))
            (new-index (make-array length :element-type 'fixnum)))
        (iterate
          (with j = 0)
          (for i from 0 below (length split-array))
          (when (eq (aref split-array i) position)
            (setf (aref new-index j) (aref old-index i))
            (incf j))
          (finally (assert (= j length) (j length))))
        (mapcar (lambda (data-matrix)
                  (funcall (data-matrix-constructor data-matrix)
                           :data (data data-matrix)
                           :index new-index))
                data-matrixes))))


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


(defun data-transpose (data)
  (declare (type data-matrix data)
           (optimize (speed 3) (safety 0)))
  (bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (iterate
      (declare (type fixnum r))
      (with result = (make-data-matrix attributes-count data-points-count
                                       :element-type (data-matrix-element-type data)))
      (for r from 0 below data-points-count)
      (iterate
        (declare (type fixnum c))
        (for c from 0 below attributes-count)
        (setf (mref result c r) (mref result r c)))
      (finally (return result)))))


(defun data-matrix-map (function data-matrix parallel &aux (data (data data-matrix)))
  (check-type data-matrix data-matrix)
  (funcall (if parallel #'lparallel:pmap #'map) nil
           (lambda (data-point) (funcall function data-point data))
           (index data-matrix))
  data-matrix)


(defun data-matrix-quasi-clone (data-matrix &rest args)
  (apply (data-matrix-constructor data-matrix)
         (append args
                 (list :data (data data-matrix)
                       :index (index data-matrix)))))
