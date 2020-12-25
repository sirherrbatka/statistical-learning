(cl:in-package #:statistical-learning.isolation-forest)


(defmethod sl.tp:split*/proxy
    (parameters/proxy
     (training-parameters isolation)
     training-state)
  (declare (optimize (debug 3)))
  (bind ((split-array #1=(~> training-state sl.mp:data-points
                             length sl.opt:make-split-array))
         (optimal-split-array #1#)
         (optimal-point nil)
         (new-depth (1+ (sl.tp:depth training-state)))
         (parallel (sl.tp:parallel training-parameters))
         ((:flet new-state (point position size))
          (sl.tp:split-training-state*
           training-parameters
           training-state
           split-array
           position
           size
           `(:depth ,new-depth)
           point))
         ((:flet subtree-impl
            (point position size &aux (state (new-state point
                                                        position
                                                        size))))
          (~>> (sl.tp:make-leaf* training-parameters)
               (sl.tp:split state)))
         ((:flet subtree (point position size &optional parallel))
          (if (and parallel (< new-depth 10))
              (lparallel:future (subtree-impl point position size))
              (subtree-impl point position size))))
    (iterate
      (repeat (repeats training-parameters))
      (for point = (sl.tp:pick-split training-state))
      (setf (sl.tp:split-point training-state) point)
      (for (values left-length right-length) =
           (sl.tp:fill-split-vector training-state
                                    split-array))
      (when (or (zerop left-length) (zerop right-length))
        (continue))
      (for score = (abs (- left-length right-length)))
      (minimize score into min)
      (when (= score min)
        (rotatef optimal-split-array split-array)
        (rotatef point optimal-point))
      (finally
       (when optimal-point
         (sl.tp:make-node 'isolation-tree
                          :size (+ left-length right-length)
                          :left-node (subtree optimal-point
                                              sl.opt:left
                                              left-length
                                              parallel)
                          :right-node (subtree optimal-point
                                               sl.opt:right
                                               right-length)
                          :point optimal-point))))))


(defmethod sl.tp:leaf-for/proxy (splitter/proxy
                                 (splitter isolation-splitter)
                                 node
                                 data
                                 index
                                 context)
  (declare (type sl.data:double-float-data-matrix data)
           (type fixnum index))
  (bind (((:labels impl
            (node depth &aux (next-depth (the fixnum (1+ depth)))))
          (declare (optimize (speed 3) (safety 0)))
          (if (sl.tp:treep node)
              (if (rightp (sl.tp:point node)
                          (attributes context)
                          index data)
                  (~> node sl.tp:right-node (impl next-depth))
                  (~> node sl.tp:left-node (impl next-depth)))
              (values node depth))))
    (impl node 0)))


(defmethod sl.tp:requires-split-p/proxy and
    (parameters/proxy
     (splitter isolation-splitter)
     (isolation isolation)
     training-state)
  (and (> (~> training-state sl.mp:data-points length)
          (sl.tp:minimal-size isolation))
       (< (sl.tp:depth training-state)
          (sl.tp:maximal-depth isolation))))


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters isolation)
                                            &rest initargs
                                            &key
                                              data
                                              c
                                              data-points
                                              attributes)
  (declare (ignore initargs))
  (make 'isolation-training-state
        :parameters parameters
        :depth 0
        :c c
        :train-data data
        :data-points data-points
        :attributes attributes))


(defmethod sl.tp:make-leaf*/proxy (training-parameters/proxy
                                   (training-parameters isolation))
  (make 'isolation-leaf))


(defmethod sl.tp:initialize-leaf/proxy (training-parameters/proxy
                                        (training-parameters isolation)
                                        training-state
                                        leaf)
  (setf (size leaf) (~> training-state sl.mp:data-points length))
  leaf)


(defmethod sl.tp:pick-split*/proxy (splitter/proxy
                                    (splitter isolation-splitter)
                                    parameters
                                    state)
  (generate-point state))


(defmethod sl.tp:fill-split-vector*/proxy
    (splitter/proxy
     (splitter isolation-splitter)
     parameters
     state
     point
     split-vector)
  (declare (type sl.data:split-vector split-vector)
           (type split-point point)
           (optimize (speed 0) (safety 3) (debug 3)))
  (bind ((data (sl.mp:train-data state))
         (data-points (sl.mp:data-points state))
         (attributes (sl.tp:attribute-indexes state)))
    (declare (type sl.data:double-float-data-matrix normals)
             (type (simple-array fixnum (*)) data-points attributes))
    (iterate
      (declare (type fixnum right-count left-count i j))
      (with right-count = 0)
      (with left-count = 0)
      (for j from 0 below (length data-points))
      (for i = (aref data-points j))
      (for rightp = (rightp point attributes i data))
      (setf (aref split-vector j) rightp)
      (if rightp (incf right-count) (incf left-count))
      (finally (return (values left-count right-count))))))


(defmethod cl-ds.utils:cloning-information append
    ((object isolation-training-state))
  '((:parameters sl.mp:training-parameters)
    (:split-point sl.tp:split-point)
    (:depth sl.tp:depth)
    (:c c)
    (:train-data sl.mp:train-data)
    (:attributes sl.tp:attribute-indexes)
    (:data-points sl.mp:data-points)))


(defmethod sl.tp:extract-predictions*/proxy (parameters/proxy
                                             (parameters isolation)
                                             state)
  (let* ((trees-count (trees-count state))
         (trees-sum (trees-sum state))
         (c (c state))
         (result (sl.data:make-data-matrix-like trees-sum)))
    (iterate
      (for i from 0 below (sl.data:data-points-count trees-sum))
      (for avg = (/ (sl.data:mref trees-sum i 0)
                    trees-count))
      (for value = (/ 1 (expt 2 (/ avg c))))
      (setf (sl.data:mref result i 0) value)
      (finally (return result)))))


(defmethod sl.tp:contribute-predictions*/proxy
    (parameters/proxy
     (parameters isolation)
     tree-model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
    (when (null state)
      (let ((c (c context))
            (data-points-count (sl.data:data-points-count data)))
        (check-type c double-float)
        (setf state (make 'isolation-prediction
                          :indexes data-points-count
                          :trees-sum (sl.data:make-data-matrix
                                      data-points-count
                                      1)
                          :c (c context)))))
  (let* ((splitter (sl.tp:splitter parameters))
         (sums (trees-sum state))
         (root (sl.tp:root tree-model)))
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (data-point)
               (bind (((:values leaf depth)
                       (~>> (sl.tp:leaf-for splitter root
                                            data data-point
                                            tree-model)
                            (funcall leaf-key))))
                 (declare (ignore leaf))
                 (incf (sl.data:mref sums data-point 0)
                       depth)))
             (sl.tp:indexes state))
    (incf (trees-count state))))


(defmethod sl.mp:make-model*/proxy
    (parameters/proxy
     (parameters isolation)
     state)
  (let* ((protoroot (sl.tp:make-leaf* parameters))
         (root (sl.tp:split state protoroot parameters/proxy)))
    (make-instance 'isolation-model
                   :parameters parameters
                   :root root
                   :attributes (sl.tp:attribute-indexes state)
                   :c (c state))))


(defmethod initialize-instance :after ((object isolation)
                                       &rest all)
  (declare (ignore all))
  (bind (((:accessors access-maximal-depth
                      access-repeats
                      access-parallel
                      access-minimal-size)
          object))
    (check-type access-maximal-depth integer)
    (check-type access-repeats integer)
    (check-type access-parallel boolean)
    (check-type access-minimal-size integer)
    (assert (> access-maximal-depth 0)
            (access-maximal-depth)
            'cl-ds:argument-value-out-of-bounds
            :value access-minimal-size
            :argument :minimal-size
            :bounds '(> minimal-size 0)
            :format-control "MAXIMAL-DEPTH should be greater then zero.")
    (assert (> access-minimal-size 0)
            (access-minimal-size)
            'cl-ds:argument-value-out-of-bounds
            :value access-minimal-size
            :argument :minimal-size
            :bounds '(> minimal-size 0)
            :format-control "MINIMAL-SIZE should be greater then zero.")))
