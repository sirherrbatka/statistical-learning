(cl:in-package #:statistical-learning.data)


(declaim (inline data))
(defun data (data-matrix)
  (declare (type data-matrix data-matrix)
           (optimize (speed 3)))
  (check-type data-matrix data-matrix)
  (if (typep data-matrix 'universal-data-matrix)
      (universal-data-matrix-data data-matrix)
      (double-float-data-matrix-data data-matrix)))


(defun copy (data-matrix)
  (check-type data-matrix data-matrix)
  (if (typep data-matrix 'universal-data-matrix)
      (make-universal-data-matrix
       :data (~> data-matrix data copy-array)
       :index (~> data-matrix index))
      (make-double-float-data-matrix
       :data (~> data-matrix data copy-array)
       :index (~> data-matrix index))))


(declaim (inline index))
(defun index (data-matrix)
  (declare (type data-matrix data-matrix)
           (optimize (speed 3)))
  (if (typep data-matrix 'universal-data-matrix)
      (universal-data-matrix-index data-matrix)
      (double-float-data-matrix-index data-matrix)))


(declaim (inline data-matrix-element-type))
(defun data-matrix-element-type (data-matrix)
  (declare (type data-matrix data-matrix)
           (optimize (speed 3)))
  (~> data-matrix data array-element-type))


(declaim (inline attributes-count))
(defun attributes-count (data-matrix)
  (declare (type data-matrix data-matrix)
           (optimize (speed 3)))
  (array-dimension (data data-matrix) 1))


(declaim (inline data-points-count))
(defun data-points-count (data-matrix)
  (declare (type data-matrix data-matrix)
           (optimize (speed 3)))
  (array-dimension (index data-matrix) 0))


(defun data-matrix-dimensions (data-matrix)
  (declare (type data-matrix data-matrix)
           (optimize (speed 3)))
  (list (array-dimension (index data-matrix) 0)
        (array-dimension (data data-matrix) 1)))


(declaim (inline missing-mask))
(defun missing-mask (data-matrix)
  (declare (type data-matrix data-matrix)
           (optimize (speed 3)))
  (if (typep data-matrix 'universal-data-matrix)
      (universal-data-matrix-missing-mask data-matrix)
      (double-float-data-matrix-missing-mask data-matrix)))


(declaim (inline mref))
(defun mref (data-matrix data-point attribute)
  (declare (type data-matrix data-matrix))
  (check-type data-matrix data-matrix)
  (let ((index (aref (index data-matrix) data-point)))
    (values (aref (data data-matrix) index attribute)
            (= (aref (missing-mask data-matrix) index attribute)
               1))))


(declaim (inline (setf mref)))
(defun (setf mref) (new-value data-matrix data-point attribute)
  (declare (type data-matrix data-matrix) (optimize (speed 3)))
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


(-> make-data-matrix (fixnum fixnum &optional t t t) data-matrix)
(defun make-data-matrix (data-points-count attributes-count
                         &optional
                           (initial-element 0.0d0)
                           (element-type 'double-float)
                           missing-mask)
  (check-type data-points-count fixnum)
  (check-type attributes-count fixnum)
  (assert (> attributes-count 0))
  (assert (> data-points-count 0))
  (econd ((eq element-type 'double-float)
          (make-double-float-data-matrix
           :data (make-array `(,data-points-count ,attributes-count)
                             :initial-element initial-element
                             :element-type 'double-float)
           :missing-mask (or missing-mask
                             (make-array `(,data-points-count ,attributes-count)
                                         :element-type 'bit
                                         :initial-element 1))
           :index (make-iota-vector data-points-count)))
         ((eq element-type t)
          (make-universal-data-matrix
           :data (make-array `(,data-points-count ,attributes-count)
                             :initial-element initial-element
                             :element-type t)
           :missing-mask (or missing-mask
                             (make-array `(,data-points-count ,attributes-count)
                                         :element-type 'bit
                                         :initial-element 1))
           :index (make-iota-vector data-points-count)))))


(defun wrap (input &optional missing-mask)
  (if (typep input 'data-matrix)
      input
      (progn
        (check-type input (or (simple-array double-float (* *))
                              (simple-array t (* *))))
        (ensure missing-mask (make-array (array-dimensions input)
                                         :element-type 'bit
                                         :initial-element 1))
        (check-type missing-mask (simple-array bit (* *)))
        (let ((element-type (array-element-type input))
              (data-points-count (array-dimension input 0)))
          (econd ((eq element-type 'double-float)
                  (make-double-float-data-matrix
                   :data input
                   :missing-mask missing-mask
                   :index (make-iota-vector data-points-count)))
                 ((eq element-type t)
                  (make-universal-data-matrix
                   :data input
                   :missing-mask missing-mask
                   :index (make-iota-vector data-points-count))))))))


(-> sample (data-matrix &key
                        (:data-points (or null vector))
                        (:attributes (or null vector)))
    data-matrix)
(defun sample (data-matrix &key data-points attributes)
  (declare (optimize (speed 3) (debug 0) (safety 0)))
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
                                         (data-matrix-element-type data-matrix)
                                         (missing-mask data-matrix)))
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
      (for (values value present) = (sl.data:mref data k attribute))
      (when present
        (setf (sl.data:mref result 0 i)
              (funcall function
                       (sl.data:mref result 0 i)
                       value))))
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
  (declare (optimize (debug 0) (safety 0) (speed 3)))
  (let* ((attributes-count (if (null attributes)
                              (attributes-count data)
                              (length attributes)))
         (data-points-count (if (null data-points)
                                (data-points-count data)
                                (length data-points)))
         (result-min (make-data-matrix 1 attributes-count))
         (result-max (make-data-matrix 1 attributes-count)))
    (declare (type fixnum attributes-count data-points-count)
             (type sl.data:double-float-data-matrix data result-min result-max))
    (iterate
      (declare (type fixnum i))
      (for i from 0 below attributes-count)
      (setf (sl.data:mref result-max 0 i) #.(coerce most-negative-single-float 'double-float)
            (sl.data:mref result-min 0 i) #.(coerce most-positive-single-float 'double-float)))
    (cl-ds.utils:cases ((null data-points))
      (iterate
        (declare (type fixnum j k1))
        (for j from 0 below data-points-count)
        (for k1 = (if (null data-points) j (aref data-points j)))
        (iterate
          (declare (type fixnum i1))
          (for i1 from 0 below attributes-count)
          (bind ((attribute (if (null attributes)
                                i1
                                (aref attributes i1)))
                 ((:values value present) (sl.data:mref data k1 attribute)))
            (when present
              (minf (sl.data:mref result-min 0 i1) value)
              (maxf (sl.data:mref result-max 0 i1) value))))
        (finally (return (cons result-min result-max)))))))


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


(-> split (list fixnum split-vector list) list)
(defun split (data-matrixes length split-array positions)
  (if (endp data-matrixes)
      nil
      (let ((old-index (~> data-matrixes first index))
            (new-index (make-array length :element-type 'fixnum)))
        (iterate
          (with j = 0)
          (for i from 0 below (length split-array))
          (when (member (aref split-array i) positions :test 'eq)
            (setf (aref new-index j) (aref old-index i))
            (incf j))
          (finally (assert (= j length) (j length))))
        (mapcar (lambda (data-matrix)
                  (funcall (data-matrix-constructor data-matrix)
                           :data (data data-matrix)
                           :missing-mask (missing-mask data-matrix)
                           :index new-index))
                data-matrixes))))


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


(defun data-matrix-map-data-points (function data-matrix parallel)
  (check-type data-matrix data-matrix)
  (funcall (if parallel #'lparallel:pmap #'map) nil
           (lambda (data-point) (funcall function data-point data-matrix))
           (~> data-matrix index length make-iota-vector))
  data-matrix)


(defun data-matrix-quasi-clone (data-matrix &rest args)
  (apply (data-matrix-constructor data-matrix)
         (append args
                 (list :data (data data-matrix)
                       :index (index data-matrix)))))
