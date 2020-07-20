(cl:in-package #:statistical-learning.data)


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
    (for i from (1- length) downto 0)
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


(-> selecting-random-attributes (fixnum fixnum)
    (-> () (simple-array fixnum (*))))
(defun selecting-random-attributes (selected-count total-count)
  (let ((iota (iota-vector total-count)))
    (lambda ()
      (take (random-in-range 1 selected-count) (reshuffle iota)))))


(defun cross-validation-folds (data-points-count number-of-folds)
  (check-type data-points-count integer)
  (check-type number-of-folds integer)
  (when (< data-points-count number-of-folds)
    (error 'cl-ds:incompatible-arguments
           :parameters '(data-points-count number-of-folds)
           :values `(,data-points-count ,number-of-folds)
           :format-control "NUMBER-OF-FOLDS can't be larger then DATA-POINTS-COUNT"))
  (when (< number-of-folds 2)
    (error 'cl-ds:invalid-argument-value
           :parameter 'number-of-folds
           :value number-of-folds
           :format-control "NUMBER-OF-FOLDS must be at least 2."))
  (let* ((indexes (~> data-points-count iota-vector reshuffle))
         (validation-size (truncate data-points-count number-of-folds))
         (train-size (- data-points-count validation-size)))
    (cl-ds:xpr (:index 0 :fold 0)
      (when (and (< fold number-of-folds))
        (let* ((validation (make-array validation-size :element-type 'fixnum))
               (train (make-array train-size :element-type 'fixnum)))
          (iterate
            (for j from 0 below validation-size)
            (for i from index below data-points-count)
            (setf (aref validation j) (aref indexes i)))
          (iterate
            (for i from 0 below index)
            (setf (aref train i) (aref indexes i)))
          (iterate
            (for j from index below train-size)
            (for i from (+ index validation-size) below data-points-count)
            (setf (aref train j) (aref indexes i)))
          (cl-ds:send-recur (list* train validation)
                            :index (+ index validation-size)
                            :fold (1+ fold)))))))
