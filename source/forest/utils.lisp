(cl:in-package #:cl-grf.forest)


(defun selecting-random-indexes (selected-count
                                 total-count)
  (check-type selected-count positive-integer)
  (check-type total-count positive-integer)
  (let ((iota (iota-vector total-count)))
    (curry #'cl-grf.data:select-random-indexes
           selected-count
           total-count
           iota)))


(defun total-support (leafs index)
  (iterate
    (for l in-vector leafs)
    (sum (cl-grf.alg:support (aref l index)))))


(-> select-random-indexes (fixnum fixnum) (simple-array fixnum (*)))
(defun select-random-indexes (selected-count total-count)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((table (make-hash-table :size total-count))
         (limit (min selected-count total-count))
         (result (make-array limit :element-type 'fixnum)))
    (iterate
      (declare (type fixnum i random-position value next-value
                     lower-bound))
      (for i from 0 below limit)
      (for lower-bound = (- total-count i))
      (for random-position = (+ i (random lower-bound)))
      (for value = (ensure (gethash i table) i))
      (for next-value = (ensure (gethash random-position table)
                          random-position))
      (unless (eql i random-position)
        (setf (gethash i table) next-value
              (gethash random-position table) value))
      (setf (aref result i) next-value)
      (finally (return result)))))
