(cl:in-package #:statistical-learning.ensemble)


(defmethod initialize-instance :after
    ((instance ensemble)
     &rest initargs)
  (declare (ignore initargs))
  (let* ((trees-count (trees-count instance))
         (tree-batch-size (tree-batch-size instance))
         (tree-attributes-count (tree-attributes-count instance))
         (tree-sample-rate (tree-sample-rate instance))
         (tree-parameters (tree-parameters instance)))
    (unless (< 0.0 tree-sample-rate 1.0)
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
                   'statistical-learning.mp:fundamental-model-parameters)
      (error 'type-error
             :expected-type 'statistical-learning.mp:fundamental-model-parameters
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
             :parameter :trees-count))))


(defmethod initialize-instance :after ((instance gradient-boost-ensemble)
                                       &rest initargs)
  (declare (ignore initargs))
  (let ((shrinkage (shrinkage instance)))
    (check-type shrinkage positive-double-float)))


(defmethod statistical-learning.mp:predict ((model ensemble-model)
                                            data
                                            &optional parallel)
  (check-type data statistical-learning.data:data-matrix)
  (let* ((trees (trees model))
         (parameters (statistical-learning.mp:parameters model))
         (tree-parameters (tree-parameters parameters))
         (result (trees-predict tree-parameters trees data parallel)))
    result))


(defmethod cl-ds.utils:cloning-information append ((state ensemble-state))
  '((:all-args all-args)))


(defmethod initialize-instance :after ((instance dynamic-weights-calculator)
                                       &rest initargs
                                       &key &allow-other-keys)
  (declare (ignore initargs))
  (let* ((weights (weights instance))
         (data-points-count (sl.data:data-points-count weights)))
    (setf (indexes instance) (sl.data:iota-vector data-points-count)
          (counts instance) (make-array `(,data-points-count 2)
                                        :element-type 'fixnum
                                        :initial-element 0))))


(defmethod update-weights ((calculator static-weights-calculator)
                           tree-parameters
                           prev-trees
                           samples)
  nil)


(defmethod update-weights ((calculator dynamic-weights-calculator)
                           (tree-parameters sl.perf:classification)
                           prev-trees
                           samples)
  (declare (type vector samples prev-trees)
           (optimize (speed 3) (safety 0)))
  (let ((indexes (indexes calculator))
        (parallel (parallel calculator))
        (train-data (train-data calculator))
        (target-data (target-data calculator))
        (weights (weights calculator))
        (counts (counts calculator)))
    (cl-ds.utils:transform #'cl-ds.alg:to-hash-table
                           samples)
    (map nil #'sl.tp:force-tree prev-trees)
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (index &aux (expected (sl.data:mref target-data
                                                         index
                                                         0)))
               (declare (type fixnum index)
                        (type double-float expected))
               (iterate
                 (for tree in-vector prev-trees)
                 (for sample in-vector samples)
                 (incf (aref counts index 0))
                 (when (gethash index sample) (next-iteration))
                 (for leaf = (sl.tp:leaf-for (sl.tp:root tree)
                                             train-data
                                             index))
                 (for predictions = (sl.tp:predictions leaf))
                 (for prediction =
                      (iterate
                        (declare (type fixnum i))
                        (for i from 0
                             below (sl.data:attributes-count predictions))
                        (finding i maximizing
                                 (sl.data:mref predictions 0 i))))
                 (when (= prediction expected)
                   (incf (aref counts index 1)))))
             indexes)
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (index &aux (total (aref counts index 0)))
               (declare (type fixnum index)
                        (type fixnum total))
               (unless (zerop total)
                 (setf (sl.data:mref weights index 0)
                       (- 1.0d0 (/ (the fixnum (aref counts index 1))
                                   total)))))
             indexes)))


(defmethod sl.mp:make-training-state ((parameters random-forest)
                                      &rest initargs
                                      &key train-data target-data weights)
  (make 'ensemble-state
        :train-data train-data
        :target-data target-data
        :weights weights
        :training-parameters parameters
        :all-args initargs))


(defmethod sl.mp:make-model* ((parameters random-forest)
                              state)
  (bind ((train-data (sl.mp:train-data state))
         (weights (sl.mp:weights state))
         (target-data (sl.mp:target-data state))
         (train-data-attributes (sl.data:attributes-count train-data))
         (tree-batch-size (tree-batch-size parameters))
         (tree-parameters (tree-parameters parameters))
         (trees-count (trees-count parameters))
         (parallel (parallel parameters))
         (tree-attributes-count (tree-attributes-count parameters))
         (trees (make-array trees-count
                            :initial-element nil))
         (samples (make-array trees-count))
         (attributes (make-array trees-count))
         (weights-calculator nil)
         (attributes-generator (sl.data:selecting-random-indexes
                                tree-attributes-count
                                train-data-attributes))
         ((:flet array-view (array &key (from 0) (to trees-count)))
          (make-array (- (min trees-count to) from)
                      :displaced-index-offset (min trees-count from)
                      :displaced-to array)))
    (statistical-learning.data:bind-data-matrix-dimensions
        ((train-data-data-points train-data-attributes train-data)
         (target-data-data-points target-data-attributes target-data))
      (setf weights (if (null weights)
                        (sl.data:make-data-matrix train-data-data-points
                                                  1
                                                  1.0d0)
                        (copy-array weights)))
      (setf weights-calculator (make (weights-calculator-class parameters)
                                     :parallel parallel
                                     :weights weights
                                     :train-data train-data
                                     :target-data target-data))
      (iterate
        (with state-initargs = '())
        (with index = 0)
        (iterate
          (while (< index trees-count))
          (for trees-view = (array-view trees
                                        :from index
                                        :to (+ index tree-batch-size)))
          (for attributes-view = (array-view attributes
                                             :from index
                                             :to (+ index tree-batch-size)))
          (map-into attributes-view attributes-generator)
          (for samples-view = (array-view samples
                                          :from index
                                          :to (+ index tree-batch-size)))
          (fit-tree-batch parameters trees-view attributes-view
                          state-initargs state
                          weights samples-view)
          (update-weights weights-calculator
                          (sl.pt:inner tree-parameters)
                          trees-view samples-view)
          (incf index tree-batch-size))
        (for swap-count = (cl-ds.utils:swap-if trees (complement #'treep)))
        (until (zerop swap-count))
        (decf index swap-count))
      (make 'random-forest-model
            :trees trees
            :parameters parameters
            :target-attributes-count target-data-attributes))))


(defmethod sl.mp:make-training-state ((parameters gradient-boost-ensemble)
                                      &rest initargs
                                      &key target-data train-data weights)
  (let* ((tree-parameters (tree-parameters parameters))
         (expected-value (sl.gbt:calculate-expected-value tree-parameters
                                                          target-data)))
    (make 'ensemble-state
          :all-args `(:expected-value ,expected-value ,@initargs)
          :train-data train-data
          :target-data target-data
          :weights weights
          :training-parameters parameters)))


(defmethod sl.mp:make-model* ((parameters gradient-boost-ensemble)
                              state)
  (bind ((train-data (sl.mp:train-data state))
         (target-data (sl.mp:target-data state))
         (train-data-attributes (sl.data:attributes-count train-data))
         (target-data-attributes (sl.data:attributes-count target-data))
         (tree-batch-size (tree-batch-size parameters))
         (tree-parameters (tree-parameters parameters))
         (trees-count (trees-count parameters))
         (samples (make-array trees-count))
         (parallel (parallel parameters))
         (tree-attributes-count (tree-attributes-count parameters))
         (trees (make-array trees-count))
         (attributes (make-array trees-count))
         (attributes-generator (sl.data:selecting-random-indexes
                                tree-attributes-count
                                train-data-attributes))
         ((:flet array-view (array &key (from 0) (to trees-count)))
          (make-array (- (min trees-count to) from)
                      :displaced-index-offset (min trees-count from)
                      :displaced-to array)))
    (iterate
      (with response = nil)
      (with contributed = nil)
      (with shrinkage = (shrinkage parameters))
      (for index
           from 0
           below trees-count
           by tree-batch-size)
      (while (< index trees-count))
      (for trees-view = (array-view trees
                                    :from index
                                    :to (+ index tree-batch-size)))
      (for attributes-view = (array-view attributes
                                         :from index
                                         :to (+ index tree-batch-size)))
      (map-into attributes-view attributes-generator)
      (for samples-view = (array-view samples
                                      :from index
                                      :to (+ index tree-batch-size)))
      (fit-tree-batch parameters trees-view attributes-view
                      `(:response ,response :shrinkage ,shrinkage)
                      state nil samples-view)
      (for new-contributed = (contribute-trees tree-parameters
                                               trees-view
                                               train-data
                                               parallel
                                               contributed))
      (setf response (sl.gbt:calculate-response tree-parameters
                                                new-contributed
                                                target-data)
            contributed new-contributed))
    (make 'gradient-boost-ensemble-model
          :trees trees
          :parameters parameters
          :target-attributes-count target-data-attributes)))


(defmethod sl.perf:performance-metric ((parameters ensemble)
                                       target
                                       predictions
                                       &key weights)
  (sl.perf:performance-metric (tree-parameters parameters)
                              target
                              predictions
                              :weights weights))


(defmethod sl.perf:average-performance-metric ((parameters ensemble)
                                               metrics)
  (sl.perf:average-performance-metric (tree-parameters parameters)
                                      metrics))


(defmethod sl.perf:errors ((parameters ensemble)
                           target
                           predictions)
  (sl.perf:errors (tree-parameters parameters)
                  target
                  predictions))
