(cl:in-package #:cl-grf.data)


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
             (aref vector (+ i (random (- length i)))))
    (finally (return vector))))


(-> select-random-indexes (fixnum fixnum) (simple-array fixnum (*)))
(defun select-random-indexes (selected-count
                              total-count
                              &optional (indexes
                                         (iota-vector total-count)))
  (check-type selected-count positive-integer)
  (check-type total-count positive-integer)
  (~>> (reshuffle indexes)
       (take selected-count)))
