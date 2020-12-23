(cl:in-package #:statistical-learning.common)


(declaim (inline random-uniform))
(-> random-uniform (double-float double-float) double-float)
(defun random-uniform (min max)
  (+ (random (- max min)) min))


(-> side-of-line (double-float double-float
                  double-float double-float
                  double-float double-float)
    double-float)
(declaim (inline side-of-line))
(defun side-of-line (p1x p1y p2x p2y p3x p3y)
  (declare (type double-float p1x p1y p2x p2y p3x p3y)
           (optimize (speed 3) (safety 0)))
  (- (* (- p1x p3x) (- p2y p3y))
     (* (- p2x p3x) (- p1y p3y))))


(defstruct gauss-random-state
  (u 0.0d0 :type double-float)
  (v 0.0d0 :type double-float)
  (phase t :type boolean)
  (lock (bt:make-lock)))


(defun gauss-random (state)
  (bind (((:structure gauss-random-state- u v phase lock)
          state))
    (bt:with-lock-held (lock)
      (prog1
          (if phase
              (progn
                (setf u (/ (+ 1.0d0 (random most-positive-fixnum))
                           (+ 2.0d0 most-positive-fixnum))
                      v (/ (random most-positive-fixnum)
                           (+ 1.0d0 most-positive-fixnum)))
                (* (sqrt (* -2.0d0 (log u)))
                   (sin (* 2.0d0 pi v))))
              (* (sqrt (* -2.0d0 (log u)))
                 (cos (* 2.0d0 pi v))))
        (setf phase (not phase))))))

