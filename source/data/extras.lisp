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


(-> selecting-random-indexes (fixnum fixnum)
    (-> () (simple-array fixnum (*))))
(defun selecting-random-indexes (selected-count total-count)
  (let ((iota (iota-vector total-count)))
    (lambda ()
      (take selected-count (reshuffle iota)))))
