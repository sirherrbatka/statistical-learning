(cl:in-package #:statistical-learning.proxy-tree)


(defclass triplet-tree (tree-proxy)
  ((%samples-count :initarg :samples-count
                   :reader samples-count)))


(defclass triplet-state (proxy-state)
  ((%positives :initarg :positives
               :reader positives)
   (%negatives :initarg :negatives
               :reader negatives)))


(defmethod cl-ds.utils:cloning-information append ((object triplet-state))
  '((:positives positives)
    (:negatives negatives)))


(defgeneric triplet-loss (splitter tree-parameters tree-state
                          split-vector left-loss right-loss))


(defmethod triplet-loss ((splitter sl.tp:distance-splitter)
                         (tree-parameters sl.dt:classification)
                         tree-state
                         split-vector
                         left-loss
                         right-loss)
  (declare (type sl.data:split-vector split-vector))
  (bind ((positives (positives tree-state))
         ((left-pivot . right-pivot) (sl.tp:split-point tree-state))
         (negatives (negatives tree-state))
         (distance-fn (sl.tp:distance-function splitter))
         (data-points (sl.mp:data-points tree-state))
         (train-data (sl.mp:train-data tree-state)))
    (iterate
     (for data-point in-vector data-points)
     (for anchor = (sl.data:mref train-data data-point 0))
     (for positive = (gethash data-point positives))
     (for negative = (gethash data-point negatives))
     (for positive-distance = (funcall distance-fn
                                       (sl.data:mref train-data positive 0)
                                       anchor))
     (for negative-distance = (funcall distance-fn
                                       (sl.data:mref train-data negative 0)
                                       anchor))
     (sum positive-distance into total-positive-difference)
     (sum negative-distance into total-negative-difference)
     (finally (return (- total-negative-difference total-positive-difference))))))


(defmethod triplet-loss ((splitter sl.tp:random-attribute-splitter)
                         (tree-parameters sl.dt:classification)
                         tree-state
                         split-vector
                         left-loss
                         right-loss)
  (declare (type sl.data:split-vector split-vector))
  (bind (((attribute . threshold) (sl.tp:split-point tree-state))
         ((:flet same-side (anchor example))
          (eq (> anchor threshold)
              (> example threshold))))
    (iterate
     (with left-sum = 0.0d0)
     (with right-sum = 0.0d0)
     (with left-scale = 0.0d0)
     (with right-scale = 0.0d0)
     (with data-points = (sl.mp:data-points tree-state))
     (with train-data = (sl.mp:train-data tree-state))
     (with positives = (positives tree-state))
     (with negatives = (negatives tree-state))
     (for i from 0 below (the fixnum (length data-points)))
     (for data-point = (aref data-points i))
     (for attribute-value = (sl.data:mref train-data data-point attribute))
     (for threshold-difference = (- threshold attribute-value))
     (for positive = (gethash data-point positives))
     (for negative = (gethash data-point negatives))
     (for positive-attribute-value =
          (sl.data:mref train-data positive attribute))
     (for negative-attribute-value =
          (sl.data:mref train-data negative attribute))
     (for positive-check = (same-side attribute-value
                                      positive-attribute-value))
     (for negative-check = (same-side attribute-value
                                      negative-attribute-value))
     (for right = (aref split-vector i))
     (if right
         (progn
           (incf right-scale 2.0d0)
           (when positive-check
             (incf right-sum))
           (unless negative-check
             (incf right-sum)))
         (progn
           (incf left-scale 2.0d0)
           (when positive-check
             (incf left-sum))
           (unless negative-check
             (incf left-sum))))
     (finally (return (values (* left-loss (- left-scale left-sum))
                              (* right-loss  (- right-scale right-sum))))))))


(defun leafs-similarity (first-leafs second-leafs)
  (declare (type (array t (*)) first-leafs second-leafs))
  (assert (array-has-fill-pointer-p first-leafs))
  (assert (array-has-fill-pointer-p second-leafs))
  (assert (= (length first-leafs) (length second-leafs)))
  (iterate
    (declare (type fixnum length i))
    (with length = (length first-leafs))
    (for i from 0 below length)
    (for l1 = (aref first-leafs i))
    (for l2 = (aref second-leafs i))
    (counting (eq l1 l2))))


(defun triplet-select (ensemble-state sample-size tree-state)
  (let* ((target-data (sl.mp:target-data ensemble-state))
         (total-data-size (sl.data:data-points-count target-data))
         (sample (~>> total-data-size sl.data:iota-vector
                      sl.data:reshuffle
                      (take sample-size)))
         (sample-size sample-size)
         (data-points (sl.mp:data-points tree-state))
         (leafs (sl.ensemble:assigned-leafs ensemble-state))
         (positives (make-hash-table))
         (negatives (make-hash-table))
         (distances (make-hash-table))
         (data-points-count (length data-points)))
    (declare (type sl.data:double-float-data-matrix target-data)
             (type simple-vector leafs)
             (type #1=(simple-array fixnum (*)) data-points sample))
    (iterate
      (declare (type fixnum i data-point)
               (type vector first-leafs)
               (type double-float target))
      (for i from 0 below data-points-count)
      (for data-point = (aref data-points i))
      (for first-leafs = (aref leafs data-point))
      (for target = (sl.data:mref target-data data-point 0))
      (iterate
        (declare (type fixnum j s))
        (for j from 0 below sample-size)
        (for s = (aref sample j))
        (for second-leafs = (aref leafs s))
        (setf (gethash s distances)
              (leafs-similarity first-leafs second-leafs)))
      (setf sample
            (the #1# (sort sample #'> :key (lambda (x) (gethash x distances)))))
      (for positive =
           (or (iterate
                (declare (type fixnum k)
                         (type double-float other-target))
                (for i from (1- sample-size) downto 0)
                (for k = (aref sample i))
                (for other-target = (sl.data:mref target-data k 0))
                (finding k such-that (= other-target target)))
               (last-elt sample)))
      (setf (gethash data-point positives) positive)
      (for negative =
           (or (iterate
                (declare (type fixnum k)
                         (type double-float other-target))
                (for i from 0 below sample-size)
                (for k = (aref sample i))
                (for other-target = (sl.data:mref target-data k 0))
                (finding k such-that (not (= other-target target))))
               (first-elt sample)))
      (setf (gethash data-point negatives) negative))
    (values positives negatives)))


(defmethod sl.tp:calculate-loss*/proxy
    ((parameters/proxy triplet-tree)
     parameters
     state
     split-array)
  ;; this works by calculating distance from both positive and negative examples, i guess but at the same time distance is dependent on the splitter. the easiest approach is to delegate actual logic to a generic function dispatched on the splitter
  (bind (((:values left-loss right-loss) (call-next-method)))
    (triplet-loss (sl.tp:splitter parameters)
                  parameters state split-array
                  left-loss
                  right-loss)))


(defmethod sl.mp:sample-training-state*/proxy
    ((parameters/proxy triplet-tree)
     parameters
     state
     &rest args
     &key &allow-other-keys)
  (let* ((inner-sample (apply
                        #'sl.mp:sample-training-state*/proxy
                        (sl.common:next-proxy parameters/proxy)
                        parameters
                        (inner state)
                        args)))
    (cl-ds.utils:quasi-clone* state
      :inner inner-sample)))


(defmethod sl.ensemble:make-tree-training-state/proxy
    (ensemble-parameters/proxy
     (tree-parameters/proxy triplet-tree)
     ensemble-parameters
     tree-parameters
     ensemble-state
     attributes
     data-points
     initargs)
    (bind ((tree-state (call-next-method))
           ((:values positives negatives)
            (triplet-select ensemble-state
                            (samples-count tree-parameters/proxy)
                            tree-state)))
          (make-instance 'triplet-state
                         :training-parameters tree-parameters
                         :positives positives
                         :negatives negatives
                         :inner tree-state)))


(defmethod sl.ensemble:after-tree-fitting/proxy
    (ensemble-parameters/proxy
     (tree-parameters/proxy triplet-tree)
     ensemble-parameters
     tree-parameters
     ensemble-state)
  ;; this needs to ensure that leafs are up-to-date
  (sl.ensemble:assign-leafs ensemble-state)
  (call-next-method))


(defmethod sl.mp:sample-training-state*/proxy
  ((parameters/proxy triplet-tree)
   parameters
   state
   &key data-points train-attributes target-attributes initargs
   &allow-other-keys)
  (let ((inner (inner state)))
    (~> (sl.mp:sample-training-state*/proxy
         (sl.common:next-proxy parameters/proxy)
         parameters inner
         :target-attributes target-attributes
         :train-attributes train-attributes
         :data-points data-points)
        (append initargs (cl-ds.utils:cloning-list inner))
        (apply #'make (class-of inner) _)
        (apply #'make 'triplet-state
               :positives (positives state)
               :negatives (negatives state)
               :inner _))))


(defun triplet (parameters samples-count)
  (sl.common:lift parameters 'triplet-tree
                  :samples-count samples-count))
