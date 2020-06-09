(cl:in-package #:cl-grf.forest)


(defmethod shared-initialize :after
    ((instance random-forest-parameters)
     slot-names
     &rest initargs)
  (declare (ignore initargs slot-names))
  (let* ((trees-count (trees-count instance))
         (tree-batch-size (tree-batch-size instance))
         (forest-class (forest-class instance))
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
    (unless (symbolp forest-class)
      (error 'type-error :expected-type 'symbol
                         :datum forest-class))
    (when (and parallel (cl-grf.tp:parallel tree-parameters))
      (error 'cl-ds:incompatible-arguments
             :arguments '(:parallel :tree-parameters)
             :values `(,parallel ,tree-parameters)
             :format-control "You can't request parallel creation of both the forest and the individual trees at the same time."))))


(defun trees-predict (tree-parameters trees data parallel &optional state)
  (iterate
    (for tree in-vector trees)
    (setf state (cl-grf.tp:contribute-predictions tree-parameters
                                                  tree
                                                  data
                                                  state
                                                  parallel))
    (finally (return (cl-grf.tp:extract-predictions state)))))


(defmethod cl-grf.mp:predict ((random-forest fundamental-random-forest)
                              data
                              &optional parallel)
  (check-type data cl-grf.data:data-matrix)
  (let* ((trees (trees random-forest))
         (parameters (cl-grf.mp:parameters random-forest))
         (tree-parameters (tree-parameters parameters)))
    (trees-predict tree-parameters trees data parallel)))


(defmethod weights-calculator
    ((training-parameters classification-random-forest-parameters)
     parallel
     weights
     train-data
     target-data)
  (let ((tree-parameters (tree-parameters training-parameters))
        (state nil))
    (lambda (prev-trees base)
      (let ((predictions (trees-predict tree-parameters
                                        prev-trees
                                        train-data
                                        parallel
                                        state)))
        (calculate-weights training-parameters predictions target-data base weights))
      weights)))


(defmethod weights-calculator
    ((training-parameters regression-random-forest-parameters)
     parallel
     weights
     train-data
     target-data)
  (let ((data-points-count (cl-grf.data:data-points-count train-data))
        (state nil)
        (tree-parameters (tree-parameters training-parameters)))
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
           (trees-count (trees-count parameters))
           (forest-class (forest-class parameters))
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
      (setf weights-calculator (weights-calculator parameters parallel weights
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
      (make forest-class
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


(defmethod forest-class ((parameters classification-random-forest-parameters))
  'classification-random-forest)


(defmethod forest-class ((parameters regression-random-forest-parameters))
  'regression-random-forest)
