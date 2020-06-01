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


(defmethod leafs-for ((forest fundamental-random-forest)
                      data
                      &optional parallel)
  (check-type data cl-grf.data:data-matrix)
  (leafs-for* (trees forest)
              (attributes forest)
              data
              parallel))


(defmethod predictions-from-leafs ((forest classification-random-forest)
                                   leafs
                                   &optional parallel)
  (classification-predictions-from-leafs* leafs
                                          parallel))


(defmethod cl-grf.mp:predict ((random-forest fundamental-random-forest)
                              data
                              &optional parallel)
  (check-type data cl-grf.data:data-matrix)
  (predict random-forest data parallel))


(defmethod cl-grf.mp:make-model ((parameters random-forest-parameters)
                                 train-data
                                 target-data
                                 &optional weights)
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
           (sums nil)
           (predictions nil)
           ((:flet array-view (array &key (from 0) (to trees-count)))
            (make-array (min trees-count (- to from))
                        :displaced-index-offset (min trees-count from)
                        :displaced-to array)))
      (when (null weights)
        (setf weights (cl-grf.data:make-data-matrix train-data-data-points 1
                                                    1.0d0)))
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
        (for leafs = (leafs-for* prev-trees
                                 prev-attributes
                                 train-data
                                 parallel))
        (setf sums (classification-sums-from-leafs* leafs parallel sums))
        (ensure predictions (map 'vector #'copy-array sums))
        (classification-predictions-from-sums* sums
                                               index
                                               predictions)
        (calculate-weights predictions target-data base weights)
        (fit-tree-batch trees attributes index parameters
                        train-data target-data weights))
      (make forest-class
            :trees trees
            :target-attributes-count target-data-attributes
            :attributes attributes))))


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
