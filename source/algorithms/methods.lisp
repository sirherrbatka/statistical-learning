(cl:in-package #:cl-grf.algorithms)


(defmethod calculate-score ((training-parameters information-gain-classification)
                            split-array
                            target-data)
  (split-entropy split-array target-data))


(defmethod cl-grf.tp:split*
    ((training-parameters scored-classification)
     training-state
     leaf)
  (declare (optimize (speed 3) (safety 0)))
  (bind ((training-data (cl-grf.tp:training-data training-state))
         (trials-count (cl-grf.tp:trials-count training-parameters))
         (minimal-difference (minimal-difference training-parameters))
         (score (score leaf))
         (parallel (cl-grf.tp:parallel training-parameters))
         (target-data (cl-grf.tp:target-data training-state)))
    (declare (type fixnum trials-count)
             (type double-float score minimal-difference)
             (type boolean parallel)
             (type (simple-array double-float (* *))
                   training-data
                   target-data))
    (iterate
      (declare (type fixnum attempt left-length right-length
                     optimal-left-length optimal-right-length
                     optimal-attribute data-size)
               (type double-float
                     left-score right-score
                     minimal-score optimal-threshold))
      (with optimal-left-length = 0)
      (with optimal-right-length = 0)
      (with optimal-attribute = 0)
      (with minimal-score = most-positive-double-float)
      (with minimal-left-entropy = most-positive-double-float)
      (with minimal-right-entropy = most-positive-double-float)
      (with optimal-threshold = most-positive-double-float)
      (with data-size = (array-dimension training-data 0))
      (with split-array = (make-array data-size :element-type 'boolean
                                                :initial-element nil))
      (with optimal-array = (make-array data-size :element-type 'boolean
                                                  :initial-element nil))
      (for attempt from 0 below trials-count)
      (for (values attribute threshold) =
           (random-test training-data))
      (for (values left-length right-length) =
           (fill-split-array training-data attribute
                             threshold split-array))
      (for (values left-score right-score) =
           (calculate-score training-parameters
                            split-array target-data))
      (for split-score =
           (+ (* (/ left-length data-size) left-score)
              (* (/ right-length data-size) right-score)))
      (when (< split-score minimal-score)
        (setf minimal-score split-score
              optimal-threshold threshold
              optimal-attribute attribute
              optimal-left-length left-length
              optimal-right-length right-length
              minimal-left-entropy left-score
              minimal-right-entropy right-score)
        (rotatef split-array optimal-array))
      (finally
       (let ((difference (- (the double-float score)
                            (the double-float minimal-score))))
         (declare (type double-float difference))
         (when (< difference minimal-difference)
           (return nil))
         (return (make 'scored-tree-node
                       :left-node (make-simple-node
                                   optimal-array
                                   left-score
                                   optimal-left-length
                                   nil
                                   parallel
                                   training-state
                                   optimal-attribute)
                       :support data-size
                       :right-node (make-simple-node
                                    optimal-array
                                    right-score
                                    optimal-right-length
                                    t
                                    nil
                                    training-state
                                    optimal-attribute)
                       :score score
                       :attribute (~> training-state
                                      cl-grf.tp:attribute-indexes
                                      (aref optimal-attribute))
                       :attribute-value optimal-threshold)))))))



(defmethod cl-grf.tp:make-leaf* ((training-parameters information-gain-classification)
                                 training-state)
  (make-instance 'scored-leaf-node
                 :score (~> training-state
                            cl-grf.tp:target-data
                            total-entropy)))


(defmethod cl-grf.mp:make-model ((parameters information-gain-classification)
                                 train-data
                                 target-data)
  (check-type train-data cl-grf.data:data-matrix)
  (check-type target-data cl-grf.data:data-matrix)
  (let* ((attributes (iterate
                       (with attributes-count =
                             (cl-grf.data:attributes-count train-data))
                       (with result = (make-array attributes-count
                                                  :element-type 'fixnum
                                                  :initial-element 0))
                       (for i from 0 below attributes-count)
                       (setf (aref result i) i)
                       (finally (return result))))
         (state (make 'cl-grf.tp:fundamental-training-state
                      :training-parameters parameters
                      :attribute-indexes attributes
                      :target-data target-data
                      :training-data train-data)))
    (~>> state cl-grf.tp:make-leaf (cl-grf.tp:split state))))
