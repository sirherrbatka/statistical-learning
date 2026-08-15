(cl:in-package #:statistical-learning.htm)


(defun sift-up (heap idx key)
  (let ((parent (floor (/ (1- idx) 2))))
    (when (and (> idx 0) 
               (> (funcall key (aref heap parent)) 
                  (funcall key (aref heap idx))))
      (rotatef (aref heap idx) (aref heap parent))
      (sift-up heap parent key))))

(defun sift-down (heap idx key size)
  (let* ((left (+ (* 2 idx) 1))
         (right (+ (* 2 idx) 2))
         (smallest idx))
    (when (and (< left size) 
               (< (funcall key (aref heap left)) 
                  (funcall key (aref heap smallest))))
      (setf smallest left))
    (when (and (< right size) 
               (< (funcall key (aref heap right)) 
                  (funcall key (aref heap smallest))))
      (setf smallest right))
    (when (/= smallest idx)
      (rotatef (aref heap idx) (aref heap smallest))
      (sift-down heap smallest key size))))

(defun select-top-n (vector n &key (key #'identity))
  "Returns the vector with the n largest elements moved to the front using an in-place min-heap.
   :key allows providing a function to extract the weight of an element."
  (let ((len (length vector)))
    (cond ((<= n 0) vector)
          ((>= n len) vector)
          (t
           (dotimes (i n)
             (sift-up vector i key))
           (loop for i from n below len
                 for val = (aref vector i)
                 do (when (> (funcall key val) (funcall key (aref vector 0)))
                      (setf (aref vector 0) val)
                      (sift-down vector 0 key n)))
           vector))))
