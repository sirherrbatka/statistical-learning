(cl:in-package #:statistical-learning.tree-protocol)

(defstruct set-splitter-split-point
  (matrix-index 0 :type fixnum)
  (attribute 0 :type fixnum)
  (value 0.0d0 :type double-float)
  (sides #() :type simple-vector))


(-> wdot (sl.data:double-float-data-matrix
          (simple-array double-float (* *))
          fixnum
          fixnum
          (simple-array fixnum (*)))
    double-float)
(defun wdot (first second first-point second-point attributes)
  (declare (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0)))
  (iterate
    (declare (type fixnum i)
             (type double-float result))
    (with result = 0.0d0)
    (for i from 0 below (length attributes))
    (for attribute = (aref attributes i))
    (incf result
          (* (sl.data:mref first first-point attribute)
             (aref second second-point i)))
    (finally (return result))))


(defun split-result-loss (result state
                          &aux (data-size (~> state
                                              sl.mp:train-data
                                              sl.data:data-points-count)))
  (/ (+ (* (left-score result) (left-length result))
        (* (middle-score result) (middle-length result))
        (* (right-score result) (right-length result)))
     data-size))


(declaim (inline set-splitter-split-point-compare))
(defun set-splitter-split-point-compare (tuple point matrix-index &optional (side nil side-bound-p))
  (if side-bound-p
      (and (= (set-splitter-split-point-matrix-index point) matrix-index)
           (eql side
                (>= (aref tuple (set-splitter-split-point-matrix-index point))
                    (set-splitter-split-point-value point))))
      (and (= (set-splitter-split-point-matrix-index point) matrix-index)
           (>= (aref tuple (set-splitter-split-point-matrix-index point))
               (set-splitter-split-point-value point)))))


(defun set-splitter-split-point-side (attribute-index tuple split-point)
  (iterate
    (with desired-sides = (set-splitter-split-point-sides (first split-point)))
    (for point in split-point)
    (for side in-vector desired-sides)
    (always (set-splitter-split-point-compare tuple point attribute-index side))))


(defun set-splitter-split-point (tuple matrix-index attribute previous-points)
  (cons (make-set-splitter-split-point
         :matrix-index matrix-index
         :attribute attribute
         :value (aref tuple attribute)
         :sides (lret ((result (make-array (1+ (length previous-points))
                                           :initial-element t)))
                  (iterate
                    (for i from 1)
                    (for point in previous-points)
                    (setf (aref result i)
                          (set-splitter-split-point-compare tuple point matrix-index)))))
        previous-points))
