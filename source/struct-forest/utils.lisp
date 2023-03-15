(cl:in-package #:statistical-learning.struct-forest)


(-> euclid-distance (sl.data:double-float-data-matrix fixnum fixnum) double-float)
(defun euclid-distance (data ai bi)
  (declare (type sl.data:double-float-data-matrix data)
           (optimize (speed 3))
           (type fixnum ai bi))
  (iterate
    (declare (type fixnum i)
             (type double-float result difference))
    (with result = 0.0d0)
    (for i from 0 below (sl.data:attributes-count data))
    (for difference = (- (sl.data:mref data ai i)
                         (sl.data:mref data bi i)))
    (incf result (* difference difference))
    (finally (return (sqrt result)))))


(defun select-pivots (parameters state)
  (let* ((target-data (struct-target-data state))
         (length (sl.data:data-points-count target-data))
         (first-index (random length))
         (second-index (iterate
                         (for r = (random length))
                         (while (= r first-index))
                         (finally (return r)))))
    (declare (type sl.data:double-float-data-matrix target-data))
    (iterate
      (declare (type fixnum iterations))
      (with iterations = (relabel-iterations (relabeler parameters)))
      (with repeats = (relabel-repeats (relabeler parameters)))
      (repeat iterations)
      (for first-data-point = first-index)
      (for result =
           (iterate
             (repeat repeats)
             (for i = (random length))
             (when (= i first-index) (next-iteration))
             (for distance = (euclid-distance target-data first-data-point i))
             (finding i maximizing distance)))
      (when (= result second-index) (finish))
      (setf second-index result)
      (rotatef first-index second-index))
    (values first-index second-index)))


(defun relabel-with-pivots (state first-pivot second-pivot)
  (let* ((target-data (struct-target-data state))
         (data-points-count (sl.data:data-points-count target-data))
         (result (sl.data:make-data-matrix data-points-count 1)))
    (declare (type sl.data:double-float-data-matrix target-data result))
    (iterate
      (for i from 0 below data-points-count)
      (for first-distance = (euclid-distance target-data first-pivot i))
      (for second-distance = (euclid-distance target-data second-pivot i))
      (if (< first-distance second-distance)
          (setf (sl.data:mref result i 0) 1.0d0)
          (setf (sl.data:mref result i 0) 0.0d0))
      (finally (return result)))))
