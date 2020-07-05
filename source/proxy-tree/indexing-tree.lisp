(cl:in-package #:sl.proxy-tree)


(defclass indexing-tree (proxy-tree)
  ())


(defclass indexing-state (proxy-state)
  ((%index :initarg :index
           :accessor index))
  (:default-initargs :index 0))


(defclass indexing-leaf (sl.tp:fundamental-leaf-node)
  ((%inner :initarg :inner
           :accessor inner)
   (%index :initarg :index
           :accessor index))
  (:default-initargs
   :inner nil
   :index nil))


(defmethod sl.tp:predictions ((node indexing-leaf))
  (~> node inner sl.tp:predictions))


(defmethod cl-ds.utils:cloning-information apppend ((state indexing-state))
  '((:index index)))


(defmethod sl.tp:split-training-state* ((parameters indexing-tree)
                                        state
                                        split-array
                                        position
                                        size
                                        initargs
                                        &optional
                                          attribute-index
                                          attribute-indexes)
  (bind ((old-depth (sl.tp:depth state))
         (new-inner (sl.tp:split-training-state* (inner parameters)
                                                 (inner state)
                                                 split-array
                                                 position
                                                 size
                                                 initargs
                                                 attribute-index
                                                 attribute-indexes))
         (index (index state))
         (new-depth (sl.tp:depth new-inner)))
    (cl-ds.utils:quasi-clone* state
      :inner new-inner
      :index (if (= new-depth old-depth)
                 index
                 (dpb (if position 1 0) (byte 1 old-depth) index)))))


(defmethod sl.mp:sample-training-state* ((parameters indexing-tree)
                                         state
                                         &key
                                           data-points
                                           train-attributes
                                           initargs
                                           target-attributes)
  (cl-ds.utils:quasi-clone* state
    :inner (sl.mp:sample-training-state* (inner parameters)
                                         (inner state)
                                         :data-points data-points
                                         :train-attributes train-attributes
                                         :initargs initargs
                                         :target-attributes target-attributes)))


(defmethod sl.tp:make-leaf* ((parameters indexing-tree))
  (make 'indexing-tree))


(defmethod sl.tp:initialize-leaf ((parameters indexing-leaf)
                                  training-state
                                  leaf)
  (bind ((inner (inner training-state))
         (index training-state)
         (inner-parameters (inner parameters))
         (inner-leaf (sl.tp:make-leaf* inner-parameters)))
    (sl.tp:initialize-leaf inner-parameters
                           inner
                           inner-leaf)
    (setf (inner leaf) inner-leaf
          (index leaf) index)))


(defun indexing (parameters)
  (make 'indexing-tree
        :inner parameters))


(defmethod sl.mp:make-training-state ((parameters indexing-tree)
                                      &rest initargs
                                      &key &allow-other-keys)
  (make 'indexing-state
        :training-parameters parameters
        :inner (apply #'sl.mp:make-training-state
                      (inner parameters)
                      initargs)))


(defclass indexed-contributed-predictions ()
  ((%training-parameters :initarg :training-parameters
                         :reader sl.mp:training-parameters)
   (%indexes :initarg :indexes
             :reader indexes)
   (%results :initarg :results
             :reader results)))


(defmethod sl.tp:contribute-predictions* ((parameters indexing-tree)
                                          model
                                          data
                                          state
                                          parallel
                                          &optional (leaf-key #'index))
  (when (null state)
    (setf state (make 'indexing-state
                      :training-parameters parameters
                      :indexes (~> data
                                   sl.data:data-points-count
                                   sl.data:iota-vector)
                      :results (~> data
                                   sl.data:data-points-count
                                   make-array
                                   (map-into #'vect)))))
  (let ((root (sl.tp:root model))
        (indexes (indexes state))
        (results (results state)))
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (index)
               (let* ((leaf (sl.tp:leaf-for root data index))
                      (result (funcall leaf-key leaf)))
                 (vector-push-extend (aref results index)
                                     result)))
             indexes)
    state))


(defmethod sl.tp:extract-predictions* ((parameters indexing-tree)
                                       state)
  (results state))


(defmethod sl.mp:make-model* ((parameters indexing-tree) state)
  (make 'sl.tp:tree-model
        :parameters parameters
        :root (~>> state sl.tp:make-leaf (sl.tp:split state))))
