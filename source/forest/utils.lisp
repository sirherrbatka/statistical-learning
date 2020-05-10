(cl:in-package #:cl-grf.forest)


(-> iota-vector (fixnum) (simple-array fixnum (*)))
(defun iota-vector (total-count)
  (declare (optimize (speed 3) (safety 0))
           (type fixnum total-count))
  (lret ((result (make-array total-count :element-type 'fixnum)))
    (iterate
      (for i from 0 below total-count)
      (setf (aref result i) i))))


(-> reshuffle ((simple-array fixnum (*))) (simple-array fixnum (*)))
(defun reshuffle (vector)
  (declare (optimize (speed 3) (safety 0)))
  (iterate
    (declare (type fixnum length i))
    (with length = (length vector))
    (for i from (1- length) above 0)
    (rotatef (aref vector i)
             (aref vector (random-in-range i length)))
    (finally (return vector))))


(defun select-random-indexes (selected-count
                              total-count
                              &optional (indexes
                                         (iota-vector total-count)))
  (check-type selected-count positive-integer)
  (check-type total-count positive-integer)
  (~>> (reshuffle indexes)
       (take selected-count)))


(defun selecting-random-indexes (selected-count
                                 total-count)
  (check-type selected-count positive-integer)
  (check-type total-count positive-integer)
  (let ((iota (iota-vector total-count)))
    (curry #'select-random-indexes
           selected-count
           total-count
           iota)))


(defun total-support (leafs index)
  (iterate
    (for l in-vector leafs)
    (sum (cl-grf.alg:support (aref l index)))))
