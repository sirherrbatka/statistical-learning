(cl:in-package #:cl-grf.ensemble)


(defmethod initialize-instance :after
    ((instance ensemble-parameters)
     &rest initargs)
  (declare (ignore initargs))
  (let* ((trees-count (trees-count instance))
         (tree-batch-size (tree-batch-size instance))
         (parallel (parallel instance))
         (tree-attributes-count (tree-attributes-count instance))
         (tree-sample-rate (tree-sample-rate instance))
         (tree-parameters (tree-parameters instance)))
    (unless (< 0 tree-sample-rate 1.0)
      (error 'cl-ds:argument-value-out-of-bounds
             :value tree-sample-rate
             :bounds '(< 0 tree-sample-rate 1.0)
             :argument :tree-sample-rate))
    (unless (integerp tree-attributes-count)
      (error 'type-error
             :expected-type 'integer
             :datum tree-attributes-count))
    (unless (< 0 tree-attributes-count array-total-size-limit)
      (error 'cl-ds:argument-value-out-of-bounds
             :value tree-attributes-count
             :bounds `(< 0 :tree-attributes-count
                         ,array-total-size-limit)
             :argument :tree-attributes-count))
    (unless (typep tree-parameters
                   'cl-grf.mp:fundamental-model-parameters)
      (error 'type-error
             :expected-type 'cl-grf.mp:fundamental-model-parameters
             :datum tree-parameters))
    (unless (integerp trees-count)
      (error 'type-error :expected-type 'integer
                         :datum trees-count))
    (unless (integerp (/ trees-count tree-batch-size))
      (error 'cl-ds:incompatible-arguments
             :format-control ":TREES-COUNT is supposed to be multiple of :TREE-BATCH-SIZE"
             :parameters '(:tree-batch-size :trees-count)
             :values (list tree-batch-size trees-count)))
    (unless (< 0 trees-count array-total-size-limit)
      (error 'cl-ds:argument-value-out-of-bounds
             :value trees-count
             :bounds `(< 0 :trees-count
                         ,array-total-size-limit)
             :argument :trees-count))
    (when (and parallel (cl-grf.tp:parallel tree-parameters))
      (error 'cl-ds:incompatible-arguments
             :parameters '(:parallel :tree-parameters)
             :values `(,parallel ,tree-parameters)
             :format-control "You can't request parallel creation of both the forest and the individual trees at the same time."))))


(defmethod initialize-instance :after ((instance gradient-boost-ensemble-parameters)
                                       &rest initargs)
  (declare (ignore initargs))
  (let ((learning-rate (learning-rate instance))
        (learning-rate-change (learning-rate-change instance))
        (tree-batch-size (tree-batch-size instance))
        (trees-count (trees-count instance)))
    (check-type learning-rate double-float)
    (check-type learning-rate-change double-float)
    (unless (< (* (/ trees-count tree-batch-size)
                  learning-rate-change)
               learning-rate)
      (error 'cl-ds:incompatible-arguments
             :parameters '(:learning-rate :learning-rate-change)
             :values `(,learning-rate ,learning-rate-change)
             :format-control "LEARNING-RATE-CHANGE value implies that LEARNING-RATE will eventually goes below zero."))))


(defmethod cl-grf.mp:predict ((random-forest ensemble)
                              data
                              &optional parallel)
  (check-type data cl-grf.data:data-matrix)
  (let* ((trees (trees random-forest))
         (parameters (cl-grf.mp:parameters random-forest))
         (tree-parameters (tree-parameters parameters))
         (result (trees-predict tree-parameters trees data parallel)))
    result))


(defmethod weights-calculator
    ((training-parameters random-forest-parameters)
     (tree-parameters cl-grf.alg:classification)
     parallel
     weights
     train-data
     target-data)
  (let ((state nil))
    (lambda (prev-trees base)
      (let ((predictions (trees-predict tree-parameters
                                        prev-trees
                                        train-data
                                        parallel
                                        state)))
        (calculate-weights training-parameters predictions target-data base weights))
      weights)))


(defmethod weights-calculator
    ((training-parameters random-forest-parameters)
     (tree-parameters cl-grf.alg:regression)
     parallel
     weights
     train-data
     target-data)
  (let ((data-points-count (cl-grf.data:data-points-count train-data))
        (state nil))
    (declare (type fixnum data-points-count))
    (lambda (prev-trees base)
      (declare (ignore base))
      (iterate
        (declare (type fixnum i))
        (with predictions = (trees-predict tree-parameters
                                           prev-trees
                                           train-data
                                           parallel
                                           state))
        (for i from 0 below data-points-count)
        (setf (cl-grf.data:mref weights i 0)
              (abs (- (cl-grf.data:mref predictions i 0)
                      (cl-grf.data:mref target-data i 0)))))
      weights)))


(defmethod cl-grf.mp:make-model ((parameters random-forest-parameters)
                                 train-data
                                 target-data
                                 &key weights)
  (cl-grf.data:bind-data-matrix-dimensions
      ((train-data-data-points train-data-attributes train-data)
       (target-data-data-points target-data-attributes target-data))
    (bind ((tree-batch-size (tree-batch-size parameters))
           (tree-parameters (tree-parameters parameters))
           (trees-count (trees-count parameters))
           (parallel (parallel parameters))
           (tree-attributes-count (tree-attributes-count parameters))
           (trees (make-array trees-count))
           (attributes (make-array trees-count))
           (weights-calculator nil)
           ((:flet array-view (array &key (from 0) (to trees-count)))
            (make-array (min trees-count (- to from))
                        :displaced-index-offset (min trees-count from)
                        :displaced-to array)))
      (when (null weights)
        (setf weights (cl-grf.data:make-data-matrix train-data-data-points 1
                                                    1.0d0)))
      (setf weights-calculator (weights-calculator parameters tree-parameters
                                                   parallel weights
                                                   train-data target-data))
      (~>> (cl-grf.data:selecting-random-indexes tree-attributes-count
                                                 train-data-attributes)
           (map-into attributes))
      (fit-tree-batch trees attributes 0 parameters
                      train-data target-data weights)
      (iterate
        (for index from tree-batch-size
             below trees-count
             by tree-batch-size)
        (for base from (1+ (truncate trees-count tree-batch-size)) downto 0)
        (for prev-trees = (array-view trees
                                      :from (- index tree-batch-size)
                                      :to index))
        (for prev-attributes = (array-view attributes
                                           :from (- index tree-batch-size)
                                           :to index))
        (funcall weights-calculator prev-trees base)
        (fit-tree-batch trees attributes index parameters
                        train-data target-data weights))
      (make 'random-forest
            :trees trees
            :parameters parameters
            :target-attributes-count target-data-attributes))))


(defmethod cl-grf.mp:make-model ((parameters gradient-boost-ensemble-parameters)
                                 train-data
                                 target-data
                                 &key)
  (cl-grf.data:bind-data-matrix-dimensions
      ((train-data-data-points train-data-attributes train-data)
       (target-data-data-points target-data-attributes target-data))
    (bind ((tree-batch-size (tree-batch-size parameters))
           (tree-parameters (tree-parameters parameters))
           (trees-count (trees-count parameters))
           (tree-sample-size (* train-data-data-points
                                (tree-sample-rate parameters)))
           (parallel (parallel parameters))
           (tree-attributes-count (tree-attributes-count parameters))
           (trees (make-array trees-count))
           (attributes (make-array trees-count))
           ((:flet array-view (array &key (from 0) (to trees-count)))
            (make-array (min trees-count (- to from))
                        :displaced-index-offset (min trees-count from)
                        :displaced-to array))
           (expected-value (~> target-data cl-ds.utils:unfold-table mean))
           ((:flet fit-tree-batch (trees attributes learning-rate predictions))
            (funcall (if parallel #'lparallel:pmap-into #'map-into)
                     trees
                     (lambda (attributes)
                       (bind ((sample (cl-grf.data:select-random-indexes
                                       tree-sample-size
                                       train-data-data-points))
                              (train (cl-grf.data:sample train-data
                                                         :data-points sample))
                              (target (cl-grf.data:sample target-data
                                                          :data-points sample))
                              (new-predictions (if (null predictions)
                                                   nil
                                                   (cl-grf.data:sample predictions
                                                                       :data-points sample))))
                         (cl-grf.mp:make-model tree-parameters
                                               train
                                               target
                                               :learning-rate learning-rate
                                               :attributes attributes
                                               :predictions new-predictions
                                               :expected-value expected-value)))
                     attributes)))
      (~>> (cl-grf.data:selecting-random-indexes tree-attributes-count
                                                 train-data-attributes)
           (map-into attributes))
      (iterate
        (with learning-rate = (learning-rate parameters))
        (with learning-rate-change = (learning-rate-change parameters))
        (with predictions = nil)
        (with state = nil)
        (for index from 0
             below trees-count
             by tree-batch-size)
        (for trees-view = (array-view trees
                                      :from index
                                      :to (+ index tree-batch-size)))
        (for attributes-view = (array-view attributes
                                           :from index
                                           :to (+ index tree-batch-size)))
        (fit-tree-batch trees-view attributes-view learning-rate predictions)
        (for (values new-predictions new-state) = (trees-predict tree-parameters
                                                                 trees-view
                                                                 train-data
                                                                 parallel
                                                                 state))
        (decf learning-rate learning-rate-change)
        (setf predictions new-predictions
              state new-state))
      (make 'gradient-boost-ensemble
            :trees trees
            :parameters parameters
            :target-attributes-count target-data-attributes))))


(defmethod cl-grf.performance:performance-metric ((parameters random-forest-parameters)
                                                  target
                                                  predictions)
  (cl-grf.performance:performance-metric (tree-parameters parameters)
                                         target
                                         predictions))


(defmethod cl-grf.performance:average-performance-metric ((parameters random-forest-parameters)
                                                          metrics)
  (cl-grf.performance:average-performance-metric (tree-parameters parameters)
                                                 metrics))


(defmethod cl-grf.performance:errors ((parameters random-forest-parameters)
                                      target
                                      predictions)
  (cl-grf.performance:errors (tree-parameters parameters)
                             target
                             predictions))
